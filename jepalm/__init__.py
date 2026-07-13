"""JEPA-LM: Joint-Embedding Predictive Language Model"""

__version__ = "0.1.0"

from .config import JEPAConfig
from .model import JEPELM
from .encoder import BidirectionalEncoder
from .target_encoder import EMATargetEncoder
from .predictor import Predictor
from .decoder import LightweightDecoder
from .loss import jepa_loss, ntp_loss, total_loss
from .masking import create_span_mask

__all__ = [
    "JEPAConfig",
    "JEPELM",
    "BidirectionalEncoder",
    "EMATargetEncoder",
    "Predictor",
    "LightweightDecoder",
    "jepa_loss",
    "ntp_loss",
    "total_loss",
    "create_span_mask",
]
