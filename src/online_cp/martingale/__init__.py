"""Conformal test martingales.

This package implements conformal test martingales for testing the exchangeability
assumption online. Martingales combine betting strategies (from ``online_cp.betting``)
with various aggregation schemes.

Submodules
----------
base
    ``ConformalTestMartingale`` abstract base class.
jumpers
    ``PluginMartingale``, ``SimpleJumper``, ``CompositeJumper``,
    ``SimpleMixtureMartingale``.
sleepers
    ``SleeperStayer``, ``SleeperDrifter``.
wrappers
    ``VilleWrapper``, ``CUSUMWrapper``, ``ShiryaevRobertsWrapper``.
legendre
    ``SimpleLegendreJumper``, ``ProductLegendreJumper``,
    ``VariationalLegendreJumper``, ``CompositeLegendreJumper``.
"""

from online_cp.betting import (
    BetaKernel as BetaKernel,
)
from online_cp.betting import (
    BetaMLE as BetaMLE,
)
from online_cp.betting import (
    BetaMoments as BetaMoments,
)
from online_cp.betting import (
    BettingStrategy as BettingStrategy,
)
from online_cp.betting import (
    ExpertAggregationStrategy as ExpertAggregationStrategy,
)
from online_cp.betting import (
    FixedStrategy as FixedStrategy,
)
from online_cp.betting import (
    GaussianKDE as GaussianKDE,
)
from online_cp.betting import (
    ParticleFilterStrategy as ParticleFilterStrategy,
)
from online_cp.betting import (
    PiecewiseConstantBetting as PiecewiseConstantBetting,
)
from online_cp.martingale.base import ConformalTestMartingale as ConformalTestMartingale
from online_cp.martingale.jumpers import (
    CompositeJumper as CompositeJumper,
)
from online_cp.martingale.jumpers import (
    PluginMartingale as PluginMartingale,
)
from online_cp.martingale.jumpers import (
    SimpleJumper as SimpleJumper,
)
from online_cp.martingale.jumpers import (
    SimpleMixtureMartingale as SimpleMixtureMartingale,
)
from online_cp.martingale.legendre import (
    STANDARD_GRID as STANDARD_GRID,
)
from online_cp.martingale.legendre import (
    CompositeLegendreJumper as CompositeLegendreJumper,
)
from online_cp.martingale.legendre import (
    ProductLegendreJumper as ProductLegendreJumper,
)
from online_cp.martingale.legendre import (
    SimpleLegendreJumper as SimpleLegendreJumper,
)
from online_cp.martingale.legendre import (
    VariationalLegendreJumper as VariationalLegendreJumper,
)
from online_cp.martingale.legendre import (
    compute_normalization_Z as compute_normalization_Z,
)
from online_cp.martingale.legendre import (
    product_betting_value as product_betting_value,
)
from online_cp.martingale.legendre import (
    shifted_legendre_poly as shifted_legendre_poly,
)
from online_cp.martingale.sleepers import (
    SleeperDrifter as SleeperDrifter,
)
from online_cp.martingale.sleepers import (
    SleeperStayer as SleeperStayer,
)
from online_cp.martingale.wrappers import (
    CUSUMWrapper as CUSUMWrapper,
)
from online_cp.martingale.wrappers import (
    ShiryaevRobertsWrapper as ShiryaevRobertsWrapper,
)
from online_cp.martingale.wrappers import (
    VilleWrapper as VilleWrapper,
)

__all__ = [
    # Base
    "ConformalTestMartingale",
    # Jumpers & mixture
    "PluginMartingale",
    "SimpleJumper",
    "CompositeJumper",
    "SimpleMixtureMartingale",
    # Sleepers
    "SleeperStayer",
    "SleeperDrifter",
    # Wrappers
    "VilleWrapper",
    "CUSUMWrapper",
    "ShiryaevRobertsWrapper",
    # Legendre jumpers
    "SimpleLegendreJumper",
    "ProductLegendreJumper",
    "VariationalLegendreJumper",
    "CompositeLegendreJumper",
    # Legendre utilities
    "STANDARD_GRID",
    "shifted_legendre_poly",
    "compute_normalization_Z",
    "product_betting_value",
    # Re-exported betting strategies (backward compat)
    "BettingStrategy",
    "BetaKernel",
    "GaussianKDE",
    "BetaMoments",
    "BetaMLE",
    "ParticleFilterStrategy",
    "FixedStrategy",
    "PiecewiseConstantBetting",
    "ExpertAggregationStrategy",
]
