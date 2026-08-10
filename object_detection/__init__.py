"""Object detection package for the University Rover Challenge.

The public inference API is re-exported here, so consumers can use either:

    from object_detection import Detector, Detection
    from object_detection.detector_api import Detector, Detection
"""

from .detector_api import Detection, Detector

__all__ = ["Detection", "Detector"]
