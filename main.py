from hydra_zen import store, zen

from dotenv import load_dotenv
import torch

from src.config import initialize_store
from src.pipeline.action import Action

load_dotenv(override=True)


def main(action: Action) -> None:
    torch.set_float32_matmul_precision("high")

    action()


if __name__ == "__main__":
    initialize_store()

    store(
        main,
        hydra_defaults=[
            "_self_",
            {"action/model": "resnet18"},
            {"action": "train"},
        ],
    )

    store.add_to_hydra_store()

    zen(main).hydra_main(config_name="main", version_base="1.3")
