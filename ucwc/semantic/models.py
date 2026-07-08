"""CLIP/RIDAS semantic codec and evaluation image sources."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import urllib.request

import torch
from torch import nn
import torch.nn.functional as F

from .clip_vit import load_vit_b16_encoder


RIDAS_DECODER_URL = "https://raw.githubusercontent.com/echojayne/RIDAS/main/clip/decoder_cifar100.pth"
RIDAS_CLASSIFIER_URL = "https://raw.githubusercontent.com/echojayne/RIDAS/main/clip/classifier.pth"
RIDAS_DECODER_PATH = Path("data/semantic_catalogs/weights/ridas/decoder_cifar100.pth")
RIDAS_CLASSIFIER_PATH = Path("data/semantic_catalogs/weights/ridas/classifier.pth")
LAYER_DECODER_DIR = Path(__file__).resolve().parent / "ckpt"


@dataclass(frozen=True, slots=True)
class ClipRidasCodecConfig:
    clip_model_name: str = "ViT-B/16"
    clip_download_root: str = "/home/users/dky/.cache/clip"
    decoder_path: str = str(RIDAS_DECODER_PATH)
    image_size: int = 224
    feature_dim: int = 512
    max_depth: int = 12
    num_classes: int = 100
    normalize_features: bool = True

    def to_dict(self) -> dict[str, int | str | bool]:
        return asdict(self)


class ClipRidasSemanticCodec(nn.Module):
    """Variable-depth CLIP ViT-B/16 encoder plus RIDAS CIFAR100 decoder."""

    def __init__(
        self,
        config: ClipRidasCodecConfig | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.config = config or ClipRidasCodecConfig()
        self.encoder = load_vit_b16_encoder(
            model_name=self.config.clip_model_name,
            download_root=self.config.clip_download_root,
            device=device,
        )
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

        self.decoder = load_semantic_decoder(self.config.decoder_path, device=device)
        self.decoder.eval()
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def encode(self, images: torch.Tensor, encoder_depth: int | None = None) -> torch.Tensor:
        depth = self.config.max_depth if encoder_depth is None else int(encoder_depth)
        normalized = preprocess_clip_tensor(images.to(self._device()), self.config.image_size)
        features = self.encoder.encode_image(normalized, encoder_depth=depth).float()
        if self.config.normalize_features:
            features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return features

    @torch.no_grad()
    def encode_layers(
        self,
        images: torch.Tensor,
        encoder_depths: list[int] | tuple[int, ...],
    ) -> dict[int, torch.Tensor]:
        normalized = preprocess_clip_tensor(images.to(self._device()), self.config.image_size)
        features_by_depth = self.encoder.encode_image_layers(normalized, encoder_depths=encoder_depths)
        if self.config.normalize_features:
            return {
                depth: features.float() / features.float().norm(dim=1, keepdim=True).clamp_min(1e-12)
                for depth, features in features_by_depth.items()
            }
        return {depth: features.float() for depth, features in features_by_depth.items()}

    @torch.no_grad()
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features.to(self._device()).float())

    def _device(self) -> torch.device:
        return next(self.decoder.parameters()).device


def ensure_ridas_weights(
    *,
    decoder_path: str | Path = RIDAS_DECODER_PATH,
    classifier_path: str | Path = RIDAS_CLASSIFIER_PATH,
    overwrite: bool = False,
) -> dict[str, str]:
    decoder = _download_if_missing(RIDAS_DECODER_URL, Path(decoder_path), overwrite=overwrite)
    classifier = _download_if_missing(RIDAS_CLASSIFIER_URL, Path(classifier_path), overwrite=overwrite)
    return {"decoder_path": str(decoder), "classifier_path": str(classifier)}


def load_clip_ridas_codec(
    *,
    clip_model_name: str = "ViT-B/16",
    clip_download_root: str = "/home/users/dky/.cache/clip",
    decoder_path: str | Path = RIDAS_DECODER_PATH,
    device: str | torch.device = "cpu",
    normalize_features: bool = True,
) -> ClipRidasSemanticCodec:
    config = ClipRidasCodecConfig(
        clip_model_name=clip_model_name,
        clip_download_root=clip_download_root,
        decoder_path=str(decoder_path),
        normalize_features=normalize_features,
    )
    return ClipRidasSemanticCodec(config, device=device)


def layer_decoder_checkpoint_path(layer: int, ckpt_dir: str | Path = LAYER_DECODER_DIR) -> Path:
    return Path(ckpt_dir).expanduser() / f"decoder_layer{int(layer)}.pth"


def load_semantic_decoder(
    decoder_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load a linear semantic decoder from a state dict or trusted saved module."""

    payload = _load_decoder_payload(Path(decoder_path).expanduser(), device=device)
    if isinstance(payload, nn.Module):
        decoder = _linear_decoder_from_state_dict(payload.state_dict())
    elif isinstance(payload, dict):
        state = payload.get("state_dict", payload)
        if isinstance(state, nn.Module):
            decoder = _linear_decoder_from_state_dict(state.state_dict())
        elif isinstance(state, dict):
            decoder = _linear_decoder_from_state_dict(state)
        else:
            raise TypeError(f"Unsupported decoder state type: {type(state)!r}")
    else:
        raise TypeError(f"Unsupported decoder checkpoint type: {type(payload)!r}")

    decoder.to(device)
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    return decoder


def decoder_linear_shape(decoder: nn.Module) -> tuple[int, int]:
    state = decoder.state_dict()
    weight_key, weight, _bias = _find_linear_weight_bias(state)
    del weight_key
    return int(weight.shape[1]), int(weight.shape[0])


def preprocess_clip_tensor(images: torch.Tensor, image_size: int) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape [batch, 3, height, width].")
    x = images.float().clamp(0.0, 1.0)
    if x.shape[-2:] != (image_size, image_size):
        x = F.interpolate(x, size=(image_size, image_size), mode="bicubic", align_corners=False)
        x = x.clamp(0.0, 1.0)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def make_image_batch(
    *,
    batch_size: int,
    image_size: int,
    num_classes: int,
    seed: int,
    dataset: str = "synthetic",
    dataset_root: str | Path = "/home/users/dky/.cache",
    download_dataset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dataset == "synthetic":
        return make_synthetic_image_batch(
            batch_size=batch_size,
            image_size=image_size,
            num_classes=num_classes,
            seed=seed,
        )
    if dataset == "cifar100":
        return make_cifar100_image_batch(
            batch_size=batch_size,
            image_size=image_size,
            seed=seed,
            dataset_root=dataset_root,
            download=download_dataset,
        )
    if dataset == "cifar10":
        return make_cifar10_image_batch(
            batch_size=batch_size,
            image_size=image_size,
            seed=seed,
            dataset_root=dataset_root,
            download=download_dataset,
        )
    raise ValueError("dataset must be 'synthetic', 'cifar10', or 'cifar100'.")


def make_synthetic_image_batch(
    *,
    batch_size: int,
    image_size: int = 224,
    num_classes: int = 100,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    images = torch.zeros(batch_size, 3, image_size, image_size, dtype=torch.float32)
    labels = torch.arange(batch_size, dtype=torch.long) % num_classes
    grid = torch.linspace(0.0, 1.0, image_size)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")

    for index, label_tensor in enumerate(labels):
        label = int(label_tensor.item())
        channel = label % 3
        pattern_id = label % 5
        if pattern_id == 0:
            pattern = (xx > 0.25 + 0.01 * (label % 20)).float()
        elif pattern_id == 1:
            pattern = (yy > 0.25 + 0.01 * (label % 20)).float()
        elif pattern_id == 2:
            pattern = ((xx - yy).abs() < 0.10 + 0.002 * (label % 20)).float()
        elif pattern_id == 3:
            center_x = 0.25 + 0.02 * (label % 20)
            center_y = 0.35 + 0.015 * (label % 20)
            pattern = (((xx - center_x) ** 2 + (yy - center_y) ** 2) < 0.04).float()
        else:
            pattern = (torch.sin((label % 11 + 1) * torch.pi * xx) > 0).float()
        images[index, channel] = pattern
        images[index] += 0.01 * torch.randn((3, image_size, image_size), generator=generator)

    return images.clamp(0.0, 1.0), labels


def make_cifar100_image_batch(
    *,
    batch_size: int,
    image_size: int,
    seed: int,
    dataset_root: str | Path,
    download: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        return _make_cifar100_image_batch_hf(
            batch_size=batch_size,
            image_size=image_size,
            seed=seed,
            dataset_root=dataset_root,
        )
    except Exception:
        if not download:
            raise
    from torchvision.datasets import CIFAR100
    from torchvision.transforms import Compose, Resize, ToTensor

    transform = Compose([Resize((image_size, image_size)), ToTensor()])
    dataset = CIFAR100(
        root=str(Path(dataset_root).expanduser()),
        train=False,
        transform=transform,
        download=download,
    )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:batch_size].tolist()
    images: list[torch.Tensor] = []
    labels: list[int] = []
    for index in indices:
        image, label = dataset[index]
        images.append(image)
        labels.append(int(label))
    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)


def make_cifar10_image_batch(
    *,
    batch_size: int,
    image_size: int,
    seed: int,
    dataset_root: str | Path,
    download: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        return _make_cifar10_image_batch_hf(
            batch_size=batch_size,
            image_size=image_size,
            seed=seed,
            dataset_root=dataset_root,
        )
    except Exception:
        if not download:
            raise
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import Compose, Resize, ToTensor

    transform = Compose([Resize((image_size, image_size)), ToTensor()])
    dataset = CIFAR10(
        root=str(Path(dataset_root).expanduser()),
        train=False,
        transform=transform,
        download=download,
    )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:batch_size].tolist()
    images: list[torch.Tensor] = []
    labels: list[int] = []
    for index in indices:
        image, label = dataset[index]
        images.append(image)
        labels.append(int(label))
    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)


def _make_cifar100_image_batch_hf(
    *,
    batch_size: int,
    image_size: int,
    seed: int,
    dataset_root: str | Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    from datasets import load_dataset
    from torchvision.transforms import Compose, Resize, ToTensor

    root = Path(dataset_root).expanduser()
    cache_dir = root / "huggingface" / "datasets"
    if not cache_dir.exists():
        cache_dir = root
    dataset = load_dataset(
        "cifar100",
        split="test",
        cache_dir=str(cache_dir),
        download_mode="reuse_cache_if_exists",
    )
    transform = Compose([Resize((image_size, image_size)), ToTensor()])
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:batch_size].tolist()
    images: list[torch.Tensor] = []
    labels: list[int] = []
    for index in indices:
        row = dataset[int(index)]
        images.append(transform(row["img"]))
        labels.append(int(row["fine_label"]))
    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)


def _make_cifar10_image_batch_hf(
    *,
    batch_size: int,
    image_size: int,
    seed: int,
    dataset_root: str | Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    from datasets import load_dataset
    from torchvision.transforms import Compose, Resize, ToTensor

    root = Path(dataset_root).expanduser()
    cache_dir = root / "huggingface" / "datasets"
    if not cache_dir.exists():
        cache_dir = root
    dataset = load_dataset(
        "cifar10",
        split="test",
        cache_dir=str(cache_dir),
        download_mode="reuse_cache_if_exists",
    )
    transform = Compose([Resize((image_size, image_size)), ToTensor()])
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:batch_size].tolist()
    images: list[torch.Tensor] = []
    labels: list[int] = []
    for index in indices:
        row = dataset[int(index)]
        images.append(transform(row["img"]))
        labels.append(int(row["label"]))
    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)


def _download_if_missing(url: str, path: Path, *, overwrite: bool) -> Path:
    target = path.expanduser()
    if target.exists() and not overwrite:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        target.write_bytes(response.read())
    return target


def _load_decoder_payload(path: Path, *, device: str | torch.device) -> object:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        # Layer checkpoints in ucwc/semantic/ckpt are local trusted files saved as
        # full nn.Linear modules, which PyTorch's weights_only loader rejects.
        return torch.load(path, map_location=device, weights_only=False)


def _linear_decoder_from_state_dict(state_dict: dict[str, torch.Tensor]) -> nn.Linear:
    _weight_key, weight, bias = _find_linear_weight_bias(state_dict)
    out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
    decoder = nn.Linear(in_features, out_features, bias=bias is not None)
    state = {"weight": weight.detach().cpu().float()}
    if bias is not None:
        state["bias"] = bias.detach().cpu().float()
    decoder.load_state_dict(state, strict=True)
    return decoder


def _find_linear_weight_bias(
    state_dict: dict[str, torch.Tensor],
) -> tuple[str, torch.Tensor, torch.Tensor | None]:
    tensors = {key: value for key, value in state_dict.items() if torch.is_tensor(value)}
    for weight_key, weight in tensors.items():
        if not weight_key.endswith("weight") or weight.ndim != 2:
            continue
        prefix = weight_key[: -len("weight")]
        bias = tensors.get(prefix + "bias")
        if bias is None:
            return weight_key, weight, None
        if bias.ndim == 1 and int(bias.shape[0]) == int(weight.shape[0]):
            return weight_key, weight, bias
    raise ValueError("Decoder checkpoint does not contain a linear weight/bias pair.")
