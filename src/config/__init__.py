from hydra_zen import ZenStore, store

from .model import initialize_model_store
from .pipeline.action import initialize_action_store
from .pipeline.loggers.logger import initialize_logger_store


def initialize_store() -> ZenStore:
    initialize_model_store()
    initialize_action_store()
    initialize_logger_store()

    book_store = store(group="books")
    romance_store = book_store(provider="genre: romance")
    fantasy_store = book_store(provider="genre: fantasy")

    romance_store({"title": "heartfelt"}, name="heartfelt")
    romance_store({"title": "lustfully longingness"}, name="lustfully longingness")

    fantasy_store({"title": "elvish cookbook"}, name="elvish cookbook")
    fantasy_store({"title": "dwarves can't jump"}, name="dwarves can't jump")

    return store
