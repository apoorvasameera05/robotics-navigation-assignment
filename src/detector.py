"""Model loading and shared navigation label handling."""

from ultralytics import YOLO


def normalize_name(name: str) -> str:
    """Normalize predicted labels into consistent navigation categories."""

    key = " ".join(
        str(name).strip().lower().replace("-", " ").replace("_", " ").split()
    )

    aliases = {
        # Traffic cones
        "cone": "cone",
        "traffic cone": "cone",
        "orange traffic cone": "cone",
        "road cone": "cone",
        "safety cone": "cone",

        # Road barriers and barricades
        "barrier": "barrier",
        "road barrier": "barrier",
        "road barricade": "barrier",
        "construction barrier": "barrier",
        "construction barricade": "barrier",
        "road construction barricade": "barrier",
        "traffic barrier": "barrier",
        "traffic barricade": "barrier",
        "plastic road barricade": "barrier",
        "traffic control barrier": "barrier",
        "roadside concrete barrier": "barrier",
        "barricade": "barrier",
        "roadblock": "barrier",
        "road block": "barrier",

        # Traffic signs
        "stop sign": "stop sign",
    }

    return aliases.get(key, key)


def load_detector(weights: str, open_vocabulary: bool = False):
    """Load YOLO weights and optionally configure YOLO-World prompts."""

    model = YOLO(weights)

    if open_vocabulary:
        model.set_classes([
            "traffic cone",
            "roadside concrete barrier",
            "stop sign",
        ])

    return model