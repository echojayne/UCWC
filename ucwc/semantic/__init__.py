"""Semantic codec, link simulation, and experience-table generation."""

from .channels import ChannelResult, simulate_channel
from .evaluation import SemanticEvalConfig, SemanticEvalRow, evaluate_grid, evaluate_semantic_link
from .ldpc import LDPCCode
from .link import BitTransmissionResult, SemanticTransmissionResult, transmit_bits, transmit_semantic_features
from .models import (
    ClipRidasCodecConfig,
    ClipRidasSemanticCodec,
    RIDAS_CLASSIFIER_PATH,
    RIDAS_DECODER_PATH,
    ensure_ridas_weights,
    load_clip_ridas_codec,
)
from .qam import QAMModem
from .quantization import FeatureQuantizer

__all__ = [
    "BitTransmissionResult",
    "ChannelResult",
    "ClipRidasCodecConfig",
    "ClipRidasSemanticCodec",
    "FeatureQuantizer",
    "LDPCCode",
    "QAMModem",
    "RIDAS_CLASSIFIER_PATH",
    "RIDAS_DECODER_PATH",
    "SemanticEvalConfig",
    "SemanticEvalRow",
    "SemanticTransmissionResult",
    "ensure_ridas_weights",
    "evaluate_grid",
    "evaluate_semantic_link",
    "load_clip_ridas_codec",
    "simulate_channel",
    "transmit_bits",
    "transmit_semantic_features",
]
