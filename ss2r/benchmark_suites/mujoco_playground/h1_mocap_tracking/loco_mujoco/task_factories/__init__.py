from .base import TaskFactory
from .dataset_confs import (
    AMASSDatasetConf,
    CustomDatasetConf,
    DefaultDatasetConf,
    LAFAN1DatasetConf,
)
from .imitation_factory import ImitationFactory

ImitationFactory.register()
