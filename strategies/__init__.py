"""Strategy package exposing available strategy classes.

Add new strategy classes here for automatic discovery.
"""

from .base_strategy import BaseStrategy
from .moving_average_strategy import MovingAverageStrategy
from .ml_rsi_ema_volume_strategy import MLRSIEMAVolumeStrategy
from .ml_strategy import MLStrategy

# Dictionary for convenience to fetch by name
aVAILABLE_STRATEGIES = {
    "ma": MovingAverageStrategy,
    "ml_rsi_ema_volume": MLRSIEMAVolumeStrategy,
    "ml": MLStrategy,
}

