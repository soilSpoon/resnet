from src.pipeline.action import Action

from ..auto_name_store import auto_name_store

action_store = auto_name_store(group="action")


def initialize_action_store() -> None:
    action_store(Action, name="train")
    action_store(Action, mode="test", name="test")
