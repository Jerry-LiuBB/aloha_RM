from aloha_rm.unified_api.decision import DecisionService, Detection


def test_decision_true_when_person_ratio_reaches_threshold() -> None:
    service = DecisionService(confidence_threshold=0.5)
    detections = [Detection(class_name="person", confidence=0.9, bbox=(0, 0, 50, 50))]

    result = service.person_decision(
        detections=detections,
        image_width=100,
        image_height=100,
        person_ratio_threshold=0.2,
    )

    assert result is True


def test_decision_false_when_only_low_confidence_person() -> None:
    service = DecisionService(confidence_threshold=0.5)
    detections = [Detection(class_name="person", confidence=0.4, bbox=(0, 0, 90, 90))]

    result = service.person_decision(
        detections=detections,
        image_width=100,
        image_height=100,
        person_ratio_threshold=0.1,
    )

    assert result is False
