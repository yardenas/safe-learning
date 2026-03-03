from .base import TerminalStateHandler
from .height import HeightBasedTerminalStateHandler
from .no_terminal import NoTerminalStateHandler
from .traj import RootPoseTrajTerminalStateHandler

# register all terminal state handlers
NoTerminalStateHandler.register()
HeightBasedTerminalStateHandler.register()
RootPoseTrajTerminalStateHandler.register()
