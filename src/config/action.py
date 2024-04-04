from typing import Final
from hydra_zen import store

from src.pipeline.action import Action


action_store = store(group="action")


def initialize_action_store() -> None:
    action_store(Action, name="train")
    action_store(Action, mode="test", name="test")
