from hydra_zen import ZenStore, store
from .model import initialize_model_store
from .action import initialize_action_store


def initialize_store() -> ZenStore:
    initialize_model_store()
    initialize_action_store()

    return store
