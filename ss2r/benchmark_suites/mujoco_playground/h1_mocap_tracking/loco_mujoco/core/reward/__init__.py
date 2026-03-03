from .base import Reward
from .default import (
    LocomotionReward,
    NoReward,
    TargetVelocityGoalReward,
    TargetXVelocityReward,
)
from .trajectory_based import MimicReward, TargetVelocityTrajReward
from .utils import *

# register all rewards
NoReward.register()
TargetVelocityGoalReward.register()
TargetXVelocityReward.register()
TargetVelocityTrajReward.register()
MimicReward.register()
LocomotionReward.register()
