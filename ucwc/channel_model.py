"""Radio-link generation for the UCWC prototype."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import math
from pathlib import Path
import random
from typing import Any, Protocol

from ucwc.components import BaseStation, UE
from ucwc.physics_tools import log_distance_pathloss_db, sinr_db, sinr_to_cqi


@dataclass(frozen=True)
class ChannelConfig:
    model: str = "synthetic"
    pathloss_exponent: float = 3.2
    shadowing_std_db: float = 2.5
    noise_figure_db: float = 7.0
    min_distance_m: float = 5.0

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "ChannelConfig":
        payload = payload or {}
        return cls(
            model=str(payload.get("model", cls.model)),
            pathloss_exponent=float(payload.get("pathloss_exponent", cls.pathloss_exponent)),
            shadowing_std_db=float(payload.get("shadowing_std_db", cls.shadowing_std_db)),
            noise_figure_db=float(payload.get("noise_figure_db", cls.noise_figure_db)),
            min_distance_m=float(payload.get("min_distance_m", cls.min_distance_m)),
        )


@dataclass(frozen=True)
class RadioLink:
    ue_id: str
    bs_id: str
    distance_m: float
    pathloss_db: float
    shadowing_db: float
    received_power_dbm: float
    sinr_db: float
    cqi: int
    coverage_rank: int

    def to_record(self) -> dict[str, object]:
        return asdict(self)


class ChannelModel(Protocol):
    model_name: str

    def build_links(self, base_stations: list[BaseStation], ues: list[UE]) -> list[RadioLink]:
        ...


def build_channel_model(
    payload: dict[str, Any] | None,
    *,
    seed: int,
    config_dir: str | Path,
) -> ChannelModel:
    payload = payload or {}
    config = ChannelConfig.from_mapping(payload)
    if config.model in {"synthetic", "log_distance"}:
        return SyntheticChannelModel(config=config, seed=seed)
    if config.model in {"radiomapseer", "radio_map", "dataset_radio_map"}:
        return RadioMapDatasetChannelModel.from_mapping(
            payload,
            seed=seed,
            config_dir=config_dir,
            synthetic_config=config,
        )
    raise ValueError(f"Unsupported channel.model: {config.model}")


class SyntheticChannelModel:
    """Deterministic channel proxy; not ray tracing or link-level simulation."""

    model_name = "synthetic_proxy"

    def __init__(self, config: ChannelConfig, seed: int = 7):
        self.config = config
        self.seed = seed

    def build_links(self, base_stations: list[BaseStation], ues: list[UE]) -> list[RadioLink]:
        raw_links: dict[str, list[dict[str, float | str]]] = {}
        for ue in ues:
            raw_links[ue.ue_id] = []
            for bs in base_stations:
                distance_m = max(
                    self.config.min_distance_m,
                    ue.position.distance_3d_m(bs.position),
                )
                pathloss_db = log_distance_pathloss_db(
                    distance_m=distance_m,
                    carrier_frequency_ghz=bs.carrier_frequency_ghz,
                    pathloss_exponent=self.config.pathloss_exponent,
                    min_distance_m=self.config.min_distance_m,
                )
                shadowing_db = self._stable_shadowing(ue.ue_id, bs.bs_id)
                received_power_dbm = bs.tx_power_dbm - pathloss_db + shadowing_db
                raw_links[ue.ue_id].append(
                    {
                        "ue_id": ue.ue_id,
                        "bs_id": bs.bs_id,
                        "distance_m": round(distance_m, 3),
                        "pathloss_db": round(pathloss_db, 3),
                        "shadowing_db": round(shadowing_db, 3),
                        "received_power_dbm": round(received_power_dbm, 3),
                        "bandwidth_mhz": bs.resources.bandwidth_mhz,
                    }
                )

        links: list[RadioLink] = []
        for ue in ues:
            ue_raw = raw_links[ue.ue_id]
            with_sinr = []
            for candidate in ue_raw:
                interferers = [
                    float(item["received_power_dbm"])
                    for item in ue_raw
                    if item["bs_id"] != candidate["bs_id"]
                ]
                value_db = sinr_db(
                    serving_power_dbm=float(candidate["received_power_dbm"]),
                    interference_powers_dbm=interferers,
                    bandwidth_mhz=float(candidate["bandwidth_mhz"]),
                    noise_figure_db=self.config.noise_figure_db,
                )
                record = dict(candidate)
                record["sinr_db"] = round(value_db, 3)
                with_sinr.append(record)

            ranked = sorted(with_sinr, key=lambda item: float(item["sinr_db"]), reverse=True)
            for rank, item in enumerate(ranked, start=1):
                links.append(
                    RadioLink(
                        ue_id=str(item["ue_id"]),
                        bs_id=str(item["bs_id"]),
                        distance_m=float(item["distance_m"]),
                        pathloss_db=float(item["pathloss_db"]),
                        shadowing_db=float(item["shadowing_db"]),
                        received_power_dbm=float(item["received_power_dbm"]),
                        sinr_db=float(item["sinr_db"]),
                        cqi=sinr_to_cqi(float(item["sinr_db"])),
                        coverage_rank=rank,
                    )
                )
        return links

    def _stable_shadowing(self, ue_id: str, bs_id: str) -> float:
        key = f"{self.seed}:{ue_id}:{bs_id}".encode("utf-8")
        digest = hashlib.sha256(key).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        return rng.gauss(0.0, self.config.shadowing_std_db)


@dataclass(frozen=True)
class RadioMapConfig:
    bs_id: str
    path: Path
    value_kind: str
    resolution_m_per_pixel: float
    pathloss_min_db: float
    pathloss_max_db: float
    clip_coordinates: bool = True
    y_axis: str = "down"


@dataclass
class RadioMapGrid:
    config: RadioMapConfig
    values: list[list[float]]

    @property
    def width(self) -> int:
        return len(self.values[0]) if self.values else 0

    @property
    def height(self) -> int:
        return len(self.values)

    def pathloss_db_at(self, x_m: float, y_m: float) -> float:
        raw_value = self.raw_value_at(x_m, y_m)
        return self._to_pathloss_db(raw_value)

    def raw_value_at(self, x_m: float, y_m: float) -> float:
        if not self.values or self.width == 0:
            raise ValueError(f"Empty radio map: {self.config.path}")
        x_idx = int(round(x_m / self.config.resolution_m_per_pixel))
        y_idx = int(round(y_m / self.config.resolution_m_per_pixel))
        if self.config.y_axis == "up":
            y_idx = self.height - 1 - y_idx
        if self.config.clip_coordinates:
            x_idx = min(max(x_idx, 0), self.width - 1)
            y_idx = min(max(y_idx, 0), self.height - 1)
        elif x_idx < 0 or x_idx >= self.width or y_idx < 0 or y_idx >= self.height:
            raise ValueError(
                f"Coordinate ({x_m}, {y_m}) is outside radio map {self.config.path}"
            )
        return self.values[y_idx][x_idx]

    def _to_pathloss_db(self, raw_value: float) -> float:
        kind = self.config.value_kind
        if kind == "pathloss_db":
            return float(raw_value)
        if kind == "rss_dbm":
            # RSS maps are converted to pathloss later from Tx power. Keep the
            # sign convention explicit by returning a negative sentinel.
            return float("nan")
        if kind == "scaled_pathloss":
            span = self.config.pathloss_max_db - self.config.pathloss_min_db
            return self.config.pathloss_min_db + (float(raw_value) / 255.0) * span
        if kind == "scaled_gain":
            span = self.config.pathloss_max_db - self.config.pathloss_min_db
            return self.config.pathloss_max_db - (float(raw_value) / 255.0) * span
        raise ValueError(f"Unsupported radio-map value_kind: {kind}")


class RadioMapDatasetChannelModel:
    """Channel model backed by public ray-tracing radio-map grids.

    This adapter intentionally only uses dense pathloss/RSS maps as a structured
    radio-state source. It is not a full electromagnetic solver.
    """

    model_name = "radiomapseer_radio_map"

    def __init__(
        self,
        *,
        maps_by_bs_id: dict[str, RadioMapGrid],
        config: ChannelConfig,
        fallback: SyntheticChannelModel | None = None,
    ):
        self.maps_by_bs_id = maps_by_bs_id
        self.config = config
        self.fallback = fallback

    @classmethod
    def from_mapping(
        cls,
        payload: dict[str, Any],
        *,
        seed: int,
        config_dir: str | Path,
        synthetic_config: ChannelConfig,
    ) -> "RadioMapDatasetChannelModel":
        radio_map_cfg = payload.get("radio_map") or {}
        if not isinstance(radio_map_cfg, dict):
            raise ValueError("channel.radio_map must be a mapping.")
        root = _resolve_path(
            radio_map_cfg.get("dataset_root", "."),
            config_dir=config_dir,
        )
        defaults = {
            "value_kind": str(radio_map_cfg.get("value_kind", "auto")),
            "resolution_m_per_pixel": float(radio_map_cfg.get("resolution_m_per_pixel", 1.0)),
            "pathloss_min_db": float(radio_map_cfg.get("pathloss_min_db", 45.0)),
            "pathloss_max_db": float(radio_map_cfg.get("pathloss_max_db", 160.0)),
            "clip_coordinates": bool(radio_map_cfg.get("clip_coordinates", True)),
            "y_axis": str(radio_map_cfg.get("y_axis", "down")),
        }
        map_specs = _normalize_map_specs(radio_map_cfg, root=root)
        if not map_specs:
            raise ValueError(
                "channel.model=radiomapseer requires channel.radio_map.maps "
                "or channel.radio_map.map_path_pattern."
            )
        maps_by_bs_id = {}
        for spec in map_specs:
            bs_id = str(spec["bs_id"])
            path = _resolve_path(spec["path"], config_dir=root)
            value_kind = str(spec.get("value_kind", defaults["value_kind"]))
            if value_kind == "auto":
                value_kind = _infer_value_kind(path)
            map_config = RadioMapConfig(
                bs_id=bs_id,
                path=path,
                value_kind=value_kind,
                resolution_m_per_pixel=float(
                    spec.get("resolution_m_per_pixel", defaults["resolution_m_per_pixel"])
                ),
                pathloss_min_db=float(spec.get("pathloss_min_db", defaults["pathloss_min_db"])),
                pathloss_max_db=float(spec.get("pathloss_max_db", defaults["pathloss_max_db"])),
                clip_coordinates=bool(spec.get("clip_coordinates", defaults["clip_coordinates"])),
                y_axis=str(spec.get("y_axis", defaults["y_axis"])),
            )
            maps_by_bs_id[bs_id] = RadioMapGrid(
                config=map_config,
                values=_load_radio_map_values(path),
            )
        fallback = None
        if bool(radio_map_cfg.get("fallback_to_synthetic", True)):
            fallback = SyntheticChannelModel(config=synthetic_config, seed=seed)
        return cls(maps_by_bs_id=maps_by_bs_id, config=synthetic_config, fallback=fallback)

    def build_links(self, base_stations: list[BaseStation], ues: list[UE]) -> list[RadioLink]:
        raw_links: dict[str, list[dict[str, float | str]]] = {}
        for ue in ues:
            raw_links[ue.ue_id] = []
            for bs in base_stations:
                raw_links[ue.ue_id].append(self._raw_link(bs=bs, ue=ue))

        links: list[RadioLink] = []
        for ue in ues:
            ue_raw = raw_links[ue.ue_id]
            with_sinr = []
            for candidate in ue_raw:
                interferers = [
                    float(item["received_power_dbm"])
                    for item in ue_raw
                    if item["bs_id"] != candidate["bs_id"]
                ]
                value_db = sinr_db(
                    serving_power_dbm=float(candidate["received_power_dbm"]),
                    interference_powers_dbm=interferers,
                    bandwidth_mhz=float(candidate["bandwidth_mhz"]),
                    noise_figure_db=self.config.noise_figure_db,
                )
                record = dict(candidate)
                record["sinr_db"] = round(value_db, 3)
                with_sinr.append(record)

            ranked = sorted(with_sinr, key=lambda item: float(item["sinr_db"]), reverse=True)
            for rank, item in enumerate(ranked, start=1):
                links.append(
                    RadioLink(
                        ue_id=str(item["ue_id"]),
                        bs_id=str(item["bs_id"]),
                        distance_m=float(item["distance_m"]),
                        pathloss_db=float(item["pathloss_db"]),
                        shadowing_db=float(item["shadowing_db"]),
                        received_power_dbm=float(item["received_power_dbm"]),
                        sinr_db=float(item["sinr_db"]),
                        cqi=sinr_to_cqi(float(item["sinr_db"])),
                        coverage_rank=rank,
                    )
                )
        return links

    def _raw_link(self, *, bs: BaseStation, ue: UE) -> dict[str, float | str]:
        grid = self.maps_by_bs_id.get(bs.bs_id)
        if grid is None:
            if self.fallback is None:
                raise ValueError(f"No radio map configured for base station {bs.bs_id}")
            return self._fallback_raw_link(bs=bs, ue=ue)
        distance_m = max(self.config.min_distance_m, ue.position.distance_3d_m(bs.position))
        pathloss_db = grid.pathloss_db_at(ue.position.x_m, ue.position.y_m)
        if math.isnan(pathloss_db):
            raw_rss = grid.raw_value_at(ue.position.x_m, ue.position.y_m)
            received_power_dbm = raw_rss
            pathloss_db = bs.tx_power_dbm - received_power_dbm
        else:
            received_power_dbm = bs.tx_power_dbm - pathloss_db
        return {
            "ue_id": ue.ue_id,
            "bs_id": bs.bs_id,
            "distance_m": round(distance_m, 3),
            "pathloss_db": round(pathloss_db, 3),
            "shadowing_db": 0.0,
            "received_power_dbm": round(received_power_dbm, 3),
            "bandwidth_mhz": bs.resources.bandwidth_mhz,
        }

    def _fallback_raw_link(self, *, bs: BaseStation, ue: UE) -> dict[str, float | str]:
        assert self.fallback is not None
        distance_m = max(self.config.min_distance_m, ue.position.distance_3d_m(bs.position))
        pathloss_db = log_distance_pathloss_db(
            distance_m=distance_m,
            carrier_frequency_ghz=bs.carrier_frequency_ghz,
            pathloss_exponent=self.config.pathloss_exponent,
            min_distance_m=self.config.min_distance_m,
        )
        shadowing_db = self.fallback._stable_shadowing(ue.ue_id, bs.bs_id)
        received_power_dbm = bs.tx_power_dbm - pathloss_db + shadowing_db
        return {
            "ue_id": ue.ue_id,
            "bs_id": bs.bs_id,
            "distance_m": round(distance_m, 3),
            "pathloss_db": round(pathloss_db, 3),
            "shadowing_db": round(shadowing_db, 3),
            "received_power_dbm": round(received_power_dbm, 3),
            "bandwidth_mhz": bs.resources.bandwidth_mhz,
        }


def _normalize_map_specs(radio_map_cfg: dict[str, Any], *, root: Path) -> list[dict[str, Any]]:
    maps = radio_map_cfg.get("maps")
    if isinstance(maps, list):
        return [dict(item) for item in maps]
    if isinstance(maps, dict):
        return [{"bs_id": bs_id, **dict(spec)} for bs_id, spec in maps.items()]
    pattern = radio_map_cfg.get("map_path_pattern")
    bs_ids = radio_map_cfg.get("bs_ids", [])
    if pattern and isinstance(bs_ids, list):
        return [
            {"bs_id": str(bs_id), "path": str(pattern).format(bs_id=str(bs_id))}
            for bs_id in bs_ids
        ]
    single_map = radio_map_cfg.get("map_path")
    single_bs = radio_map_cfg.get("bs_id")
    if single_map and single_bs:
        return [{"bs_id": str(single_bs), "path": str(single_map)}]
    return []


def _load_radio_map_values(path: Path) -> list[list[float]]:
    if not path.exists():
        raise FileNotFoundError(f"Radio-map file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [[float(value) for value in row] for row in csv.reader(handle) if row]
    if suffix in {".npy", ".npz"}:
        return _load_numpy_array(path)
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return _load_image_array(path)
    raise ValueError(f"Unsupported radio-map file type: {path}")


def _load_numpy_array(path: Path) -> list[list[float]]:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("Loading .npy/.npz radio maps requires numpy.") from exc
    payload = np.load(path)
    if isinstance(payload, np.lib.npyio.NpzFile):
        first_key = sorted(payload.files)[0]
        array = payload[first_key]
    else:
        array = payload
    if array.ndim != 2:
        raise ValueError(f"Radio-map array must be 2D: {path}")
    return [[float(value) for value in row] for row in array.tolist()]


def _load_image_array(path: Path) -> list[list[float]]:
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("Loading image radio maps requires Pillow.") from exc
    image = Image.open(path).convert("L")
    return [[float(value) for value in row] for row in image_to_rows(image)]


def image_to_rows(image: Any) -> list[list[int]]:
    width, height = image.size
    pixels = list(image.getdata())
    return [
        [int(pixels[y * width + x]) for x in range(width)]
        for y in range(height)
    ]


def _infer_value_kind(path: Path) -> str:
    normalized = str(path).replace("\\", "/").lower()
    if "/gain/" in normalized or "gain_" in normalized or normalized.endswith("_gain.png"):
        return "scaled_gain"
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return "scaled_pathloss"
    return "pathloss_db"


def _resolve_path(raw_path: Any, *, config_dir: str | Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    return (Path(config_dir).expanduser().resolve() / path).resolve()
