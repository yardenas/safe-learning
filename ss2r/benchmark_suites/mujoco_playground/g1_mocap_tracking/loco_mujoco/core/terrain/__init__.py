from .base import Terrain
from .dynamic import DynamicTerrain
from .rough import RoughTerrain
from .static import StaticTerrain

# register all terrains
StaticTerrain.register()
RoughTerrain.register()
