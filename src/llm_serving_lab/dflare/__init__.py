"""Educational DFlare components with a NumPy-only execution path."""

from .config import DFlareConfig
from .features import DeterministicTargetFeatures, TargetFeatureProvider
from .fusion import LayerwiseFusion, SharedFusion, rms_norm, softmax
from .proposal import BlockProposal, DFlareProposer, DFlashProposer

__all__ = [
    "BlockProposal",
    "DFlareConfig",
    "DFlareProposer",
    "DFlashProposer",
    "DeterministicTargetFeatures",
    "LayerwiseFusion",
    "SharedFusion",
    "TargetFeatureProvider",
    "rms_norm",
    "softmax",
]
