"""EDS integration owned by the ZEM Cleanliness backend."""

from backend.cleanliness_core.eds.client_v2 import EdsV2Client, EdsV2Error

__all__ = ["EdsV2Client", "EdsV2Error"]
