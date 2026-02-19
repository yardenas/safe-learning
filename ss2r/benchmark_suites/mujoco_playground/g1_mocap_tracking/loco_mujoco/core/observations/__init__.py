from .base import (
    Observation,
    ObservationContainer,
    ObservationIndexContainer,
    ObservationType,
    StatefulObservation,
)
from .goals import *

# register all goals
NoGoal.register()
GoalRandomRootVelocity.register()
GoalTrajRootVelocity.register()
GoalTrajMimic.register()
GoalTrajMimicv2.register()
