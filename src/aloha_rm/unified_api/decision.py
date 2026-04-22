from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]


class DecisionService:
    """Convert YOLO detections to a business boolean decision."""

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.confidence_threshold = confidence_threshold

    def person_decision(
        self,
        detections: list[Detection],
        image_width: float,
        image_height: float,
        person_ratio_threshold: float,
    ) -> bool:
        if image_width <= 0 or image_height <= 0:
            return False
        if not 0.0 <= person_ratio_threshold <= 1.0:
            raise ValueError("person_ratio_threshold must be in [0, 1]")

        frame_area = image_width * image_height
        max_person_ratio = 0.0
        found_person = False

        for det in detections:
            if det.class_name != "person":
                continue
            if det.confidence < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = det.bbox
            box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if box_area <= 0:
                continue

            found_person = True
            ratio = box_area / frame_area
            if ratio > max_person_ratio:
                max_person_ratio = ratio

        if not found_person:
            return False

        return max_person_ratio >= person_ratio_threshold
