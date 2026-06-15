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
    BetaMLE as BetaMLE,
    BetaMoments as BetaMoments,
    BettingStrategy as BettingStrategy,
    ExpertAggregationStrategy as ExpertAggregationStrategy,
    FixedStrategy as FixedStrategy,
    GaussianKDE as GaussianKDE,
    ParticleFilterStrategy as ParticleFilterStrategy,
    PiecewiseConstantBetting as PiecewiseConstantBetting,
)
from online_cp.martingale.base import ConformalTestMartingale as ConformalTestMartingale
from online_cp.martingale.jumpers import (
    CompositeJumper as CompositeJumper,
    PluginMartingale as PluginMartingale,
    SimpleMixtureMartingale as SimpleMixtureMartingale,
    SimpleJumper as SimpleJumper,
)
from online_cp.martingale.legendre import (
    CompositeLegendreJumper as CompositeLegendreJumper,
    ProductLegendreJumper as ProductLegendreJumper,
    STANDARD_GRID as STANDARD_GRID,
    SimpleLegendreJumper as SimpleLegendreJumper,
    VariationalLegendreJumper as VariationalLegendreJumper,
    compute_normalization_Z as compute_normalization_Z,
    product_betting_value as product_betting_value,
    shifted_legendre_poly as shifted_legendre_poly,
)
from online_cp.martingale.sleepers import (
    SleeperDrifter as SleeperDrifter,
    SleeperStayer as SleeperStayer,
)
from online_cp.martingale.wrappers import (
    CUSUMWrapper as CUSUMWrapper,
    ShiryaevRobertsWrapper as ShiryaevRobertsWrapper,
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
