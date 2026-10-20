#!/usr/bin/env python3
"""Small adapter for Landmark's native tagLandmark36h11 detector."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

try:
    from apriltag import apriltag as _AprilTag
except ImportError:  # The standard 36h11-only detector still works without it.
    _AprilTag = None


@dataclass(frozen=True)
class LandmarkTag:
    marker_id: int
    corners: np.ndarray


class LandmarkTagDetector:
    """Detect the custom Landmark family with the published AprilTag 3 fork."""

    def __init__(self, decimate: float = 1.0, threads: int = 1) -> None:
        self.available = _AprilTag is not None
        self._detector = (
            _AprilTag(
                "tagLandmark36h11",
                threads=threads,
                maxhamming=1,
                decimate=decimate,
                blur=0.0,
                refine_edges=True,
                debug=False,
            )
            if self.available
            else None
        )

    def detect(self, gray: np.ndarray) -> tuple[LandmarkTag, ...]:
        if self._detector is None:
            return ()
        if gray.ndim != 2:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        image = np.ascontiguousarray(gray, dtype=np.uint8)
        return tuple(
            LandmarkTag(
                marker_id=int(item["id"]),
                corners=np.asarray(item["lb-rb-rt-lt"], dtype=np.float32),
            )
            for item in self._detector.detect(image)
        )
