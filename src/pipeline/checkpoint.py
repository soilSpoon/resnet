from lightning.pytorch.callbacks import ModelCheckpoint


class Checkpoint(ModelCheckpoint):
    def __init__(self, dirpath: str, save_top_k: int) -> None:
        super().__init__(
            dirpath=dirpath,
            filename=f"{{epoch}}-{{val_f1:.4f}}",
            save_top_k=save_top_k,
            monitor="val_f1",
            mode="max",
        )


# 잘 안 되는 케이스를 직접 눈으로 확인하고 해결책 생각
# 이미지에 동시에 등장하는 clss간 상관관계
