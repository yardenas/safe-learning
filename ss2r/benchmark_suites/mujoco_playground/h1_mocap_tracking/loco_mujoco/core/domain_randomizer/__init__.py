from .base import DomainRandomizer
from .default import DefaultRandomizer
from .no_randomization import NoDomainRandomization

# register all domain randomizers
NoDomainRandomization.register()
DefaultRandomizer.register()
