"""Core UCWC network components."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Literal


TrafficDirection = Literal["downlink", "uplink", "bidirectional"]


@dataclass(frozen=True)
class Position:
    x_m: float
    y_m: float
    z_m: float = 1.5

    def distance_2d_m(self, other: "Position") -> float:
        return math.hypot(self.x_m - other.x_m, self.y_m - other.y_m)

    def distance_3d_m(self, other: "Position") -> float:
        dx = self.x_m - other.x_m
        dy = self.y_m - other.y_m
        dz = self.z_m - other.z_m
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def to_record(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ResourceBudget:
    max_connections: int
    bandwidth_mhz: float
    total_prb: int

    def prb_bandwidth_mhz(self) -> float:
        if self.total_prb <= 0:
            return 0.0
        return self.bandwidth_mhz / float(self.total_prb)


@dataclass(frozen=True)
class QoSRequirement:
    profile_id: str
    traffic_direction: TrafficDirection
    min_dl_mbps: float
    min_ul_mbps: float
    max_latency_ms: float
    min_reliability: float
    max_packet_loss: float
    max_jitter_ms: float
    security_level: str = "standard"
    priority: int = 5

    def to_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class BaseStation:
    bs_id: str
    position: Position
    resources: ResourceBudget
    tx_power_dbm: float = 43.0
    carrier_frequency_ghz: float = 3.5
    connected_ue_ids: list[str] = field(default_factory=list)

    def can_attach(self) -> bool:
        return len(self.connected_ue_ids) < self.resources.max_connections

    def attach(self, ue_id: str) -> bool:
        if ue_id in self.connected_ue_ids:
            return True
        if not self.can_attach():
            return False
        self.connected_ue_ids.append(ue_id)
        return True

    def reset_connections(self) -> None:
        self.connected_ue_ids.clear()

    def available_connections(self) -> int:
        return max(0, self.resources.max_connections - len(self.connected_ue_ids))

    def to_record(self) -> dict[str, object]:
        return {
            "bs_id": self.bs_id,
            **self.position.to_record(),
            "max_connections": self.resources.max_connections,
            "bandwidth_mhz": self.resources.bandwidth_mhz,
            "total_prb": self.resources.total_prb,
            "tx_power_dbm": self.tx_power_dbm,
            "carrier_frequency_ghz": self.carrier_frequency_ghz,
            "connected_ue_count": len(self.connected_ue_ids),
            "available_connections": self.available_connections(),
            "connected_ue_ids": list(self.connected_ue_ids),
        }


@dataclass
class UE:
    ue_id: str
    position: Position
    qos: QoSRequirement
    mobility_speed_kmh: float = 0.0
    mobility_class: str = "static"
    connected_bs_id: str | None = None
    max_tx_power_dbm: float = 23.0
    battery_level_pct: float = 100.0

    def to_record(self) -> dict[str, object]:
        return {
            "ue_id": self.ue_id,
            **self.position.to_record(),
            "connected_bs_id": self.connected_bs_id,
            "mobility_speed_kmh": self.mobility_speed_kmh,
            "mobility_class": self.mobility_class,
            "max_tx_power_dbm": self.max_tx_power_dbm,
            "battery_level_pct": self.battery_level_pct,
            **self.qos.to_record(),
        }
