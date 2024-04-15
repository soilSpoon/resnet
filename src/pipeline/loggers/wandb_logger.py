from lightning.pytorch import loggers


class WandbLogger(loggers.WandbLogger):
    def __init__(self, name: str, save_dir: str) -> None:
        super().__init__(
            project="resnet",
            name=name,
            save_dir=save_dir,
        )
