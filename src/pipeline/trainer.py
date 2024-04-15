import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary
from lightning.pytorch.loggers import WandbLogger


class Trainer(L.Trainer):

    def __init__(
        self,
        logger: WandbLogger,
        model_checkpoint: ModelCheckpoint,
        early_stopping_patience: int,
        batch_size: int,
        epochs: int,
    ) -> None:
        super().__init__(
            logger=logger,
            callbacks=[
                model_checkpoint,
                RichModelSummary(max_depth=-1),
                EarlyStopping(
                    monitor="val_f1", mode="max", patience=early_stopping_patience
                ),
            ],
            enable_model_summary=False,
            limit_train_batches=batch_size,
            max_epochs=epochs,
        )
