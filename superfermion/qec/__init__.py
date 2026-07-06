"""
QEC (Quantum Error Correction) module for Superfermion.

Heavy dependencies (scipy, jax) are loaded lazily on first access.
"""

import sys
from superfermion._lazy import LazyModule

__all__ = [
    "RepetitionCode", "ShorCode", "SteaneCode", "BaconShorCode", "GenericCSSCode",
    "SurfaceCode2D", "HypercubeCode4D", "ToricCode2D", "ColorCode", "HoneycombCode",
    "QECManager",
    "MWPMDecoder", "UnionFindDecoder", "BPOSD_Decoder", "NeuralDecoder",
]

_LAZY_ATTRS = {
    "RepetitionCode":    "superfermion.qec.codes.linear",
    "ShorCode":          "superfermion.qec.codes.linear",
    "SteaneCode":        "superfermion.qec.codes.linear",
    "BaconShorCode":     "superfermion.qec.codes.linear",
    "GenericCSSCode":    "superfermion.qec.codes.linear",
    "SurfaceCode2D":     "superfermion.qec.codes.topological",
    "HypercubeCode4D":   "superfermion.qec.codes.topological",
    "ToricCode2D":       "superfermion.qec.codes.topological",
    "ColorCode":         "superfermion.qec.codes.topological",
    "HoneycombCode":     "superfermion.qec.codes.topological",
    "QECManager":        "superfermion.qec.manager",
    "MWPMDecoder":       "superfermion.qec.decoders",
    "UnionFindDecoder":  "superfermion.qec.decoders",
    "BPOSD_Decoder":     "superfermion.qec.decoders",
    "NeuralDecoder":     "superfermion.qec.decoders",
}

# Replace module type so __getattr__ / __dir__ are handled by LazyModule
sys.modules[__name__].__class__ = type(
    __name__, (LazyModule,), {}
)
