from src.pipeline.loggers.wandb_logger import WandbLogger

from ...auto_name_store import auto_name_store

logger_store = auto_name_store(group="action")


def initialize_logger_store() -> None:
    # early_stopping_patience은 학습에 중요한 요소가 아님
    # 모델 종류와 사이즈 등등 모델에 대해 자세히 기록
    # 키에 대해서 설명이 필요
    # ex) bs: batch size
    # 실험이 많아지면 폴더를 깊게 분류하여 달라진 설정에 대해 한 번에 알아보기 쉽게
    logger_store(
        WandbLogger,
        name="${action.trainer.mode}-${action.model.model_size}-${action.trainer.epochs}-${action.trainer.batch_size}-${action.trainer.early_stopping_patience}",
        save_dir="",
    )
