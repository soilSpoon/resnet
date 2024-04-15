from src.models.resnet import ResNet

from .auto_name_store import auto_name_store

INPUT_CHANNELS = 3
NUM_CLASSES = 40
MODEL_CONFIG = {
    0: ([1, 1], False),
    18: ([2, 2, 2, 2], False),
    34: ([3, 4, 6, 3], False),
    50: ([3, 4, 6, 3], True),
    101: ([3, 4, 23, 3], True),
    152: ([3, 8, 36, 3], True),
}

model_store = auto_name_store(group="action/model")


def initialize_model_store():
    for key in MODEL_CONFIG:
        num_layers, use_bottleneck = MODEL_CONFIG[key]

        model_parameters = {
            "input_size": INPUT_CHANNELS,
            "num_layers": num_layers,
            "num_classes": NUM_CLASSES,
            "use_bottleneck": use_bottleneck,
        }

        model_store(ResNet, **model_parameters, model_size=key, name=f"resnet{key}")
