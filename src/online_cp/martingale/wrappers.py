"""Change-point detection wrappers for conformal test martingales.

A conformal test martingale gives evidence against exchangeability, but turning
that evidence into a change-point *alarm* requires a stopping rule. These
wrappers each accumulate the martingale's increments differently:

- :class:`VilleWrapper` — raises an alarm the first time the (running maximum of
  the) martingale crosses a fixed threshold. Anytime-valid via Ville's
  inequality; best when the change, if any, is assumed to persist from a single
  unknown time and you want a simple global false-alarm guarantee.
- :class:`CUSUMWrapper` — a CUSUM statistic ($\\max$ over restart points) that
  resets after dips, making it more sensitive to a *late* change in a long
  stream.
- :class:`ShiryaevRobertsWrapper` — the Shiryaev–Roberts statistic (a *sum* over
  restart points), which is always at least the CUSUM statistic and gives a
  different power / false-alarm trade-off, often preferred for minimising
  detection delay.

See [ALRW2 §8.3] (Vovk, Gammerman & Shafer, 2022).
"""
from __future__ import annotations

import numpy as np


class VilleWrapper:
    """Ville's inequality procedure for change-point detection.

    The simplest test based on a conformal test martingale: reject the
    exchangeability hypothesis when the running maximum of the martingale
    exceeds a threshold c. By Ville's inequality:

        P(∃n : S_n >= c) <= 1/c

    So threshold c = 20 gives a 5% significance level, c = 100 gives 1%, etc.

    Parameters
    ----------
    martingale : ConformalTestMartingale
        The underlying martingale to wrap.
    threshold : float
        Default alarm threshold (default 20, i.e. 5% significance).

    References
    ----------
    Vovk, Gammerman & Shafer (2022). *Algorithmic Learning in a Random World*,
    2nd edition, §8.4.1 (The Ville Procedure). Cambridge University Press.

    Examples
    --------
    >>> from online_cp.martingale import SimpleJumper, VilleWrapper
    >>> sj = SimpleJumper(J=0.1)
    >>> ville = VilleWrapper(sj, threshold=20)
    >>> for _ in range(10):
    ...     ville.update(0.5)
    >>> bool(ville.rejected)
    False
    """

    def __init__(self, martingale, threshold=20):
        self.martingale = martingale
        self.threshold = threshold
        self._log_max = 0.0
        self._n = 0
        self._rejection_time = None

    @property
    def log_max(self):
        """Log of the running maximum of the martingale."""
        return self._log_max

    @property
    def max(self):
        """Running maximum of the martingale."""
        return np.exp(self._log_max)

    @property
    def rejected(self):
        """Whether the exchangeability hypothesis has been rejected."""
        return self._log_max >= np.log(self.threshold)

    @property
    def rejection_time(self):
        """Step at which the hypothesis was first rejected, or None."""
        return self._rejection_time

    def update(self, p: float) -> None:
        """Update the inner martingale and track the running maximum."""
        self.martingale.update(p)
        self._n += 1
        if self.martingale.logM > self._log_max:
            self._log_max = self.martingale.logM
        if self._rejection_time is None and self._log_max >= np.log(self.threshold):
            self._rejection_time = self._n

    def alarm(self, threshold=None):
        """Check whether max(S_n) exceeds the threshold.

        Parameters
        ----------
        threshold : float or None
            Override threshold. If None, uses the threshold set at construction.

        Returns
        -------
        bool
            True if the running maximum exceeds the threshold.
        """
        t = threshold if threshold is not None else self.threshold
        return self._log_max >= np.log(t)


class CUSUMWrapper:
    """CUSUM change-detection wrapper for any conformal test martingale.

    Computes the Page CUSUM statistic as the ratio of the current martingale
    value to its running minimum:

        gamma_n = S_n / min_{i <= n} S_i

    In log-space: log(gamma_n) = logM_n - min_{i <= n} logM_i

    This removes any accumulated "debt" from an initial in-control period,
    giving faster detection after the change-point. Optionally accepts a linear
    barrier for controlling the false alarm rate over long horizons.

    Parameters
    ----------
    martingale : ConformalTestMartingale
        The underlying martingale to wrap.
    barrier_slope : float or None
        If not None, the alarm threshold grows linearly as barrier_slope * n.

    References
    ----------
    Vovk, Gammerman & Shafer (2022). *Algorithmic Learning in a Random World*,
    2nd edition, §8.3. Cambridge University Press.

    Examples
    --------
    >>> from online_cp.martingale import SimpleJumper, CUSUMWrapper
    >>> sj = SimpleJumper(J=0.01)
    >>> cusum = CUSUMWrapper(sj)
    >>> for _ in range(10):
    ...     cusum.update(0.5)
    >>> bool(cusum.gamma >= 1.0)  # gamma is always >= 1 (since S_n >= min S_i is not guaranteed)
    True
    """

    def __init__(self, martingale, barrier_slope=None):
        self.martingale = martingale
        self.barrier_slope = barrier_slope
        self._log_min = 0.0  # log(S_0) = 0
        self._n = 0
        self._log_gamma_values = [0.0]

    @property
    def gamma(self):
        """Current CUSUM statistic."""
        return np.exp(self._log_gamma_values[-1])

    @property
    def log_gamma(self):
        """Current log CUSUM statistic."""
        return self._log_gamma_values[-1]

    @property
    def cusum_values(self):
        """All CUSUM statistic values."""
        return np.exp(self._log_gamma_values)

    @property
    def log_cusum_values(self):
        """All log CUSUM statistic values."""
        return self._log_gamma_values

    def update(self, p: float) -> None:
        """Update the inner martingale and recompute the CUSUM statistic."""
        self.martingale.update(p)
        self._n += 1

        logM = self.martingale.logM
        if logM < self._log_min:
            self._log_min = logM

        log_gamma = logM - self._log_min
        self._log_gamma_values.append(log_gamma)

    def alarm(self, threshold):
        """Check whether gamma_n exceeds the threshold (optionally with barrier).

        Parameters
        ----------
        threshold : float
            The alarm threshold. If barrier_slope is set, the effective
            threshold at step n is threshold + barrier_slope * n.

        Returns
        -------
        bool
            True if gamma_n exceeds the (possibly time-varying) threshold.
        """
        effective = threshold
        if self.barrier_slope is not None:
            effective = threshold + self.barrier_slope * self._n
        return self.gamma > effective


class ShiryaevRobertsWrapper:
    """Shiryaev-Roberts change-detection wrapper for any conformal test martingale.

    Computes the Shiryaev-Roberts statistic as:

        R_n = sum_{i=1}^{n} S_n / S_i

    In log-space: R_n = sum_{i=1}^{n} exp(logM_n - logM_{i-1})

    This is always >= the CUSUM statistic (sum >= max), giving a slightly
    different power/false-alarm trade-off.

    Parameters
    ----------
    martingale : ConformalTestMartingale
        The underlying martingale to wrap.

    References
    ----------
    Vovk, Gammerman & Shafer (2022). *Algorithmic Learning in a Random World*,
    2nd edition, §8.3. Cambridge University Press.

    Examples
    --------
    >>> from online_cp.martingale import SimpleJumper, ShiryaevRobertsWrapper
    >>> sj = SimpleJumper(J=0.01)
    >>> sr = ShiryaevRobertsWrapper(sj)
    >>> for _ in range(10):
    ...     sr.update(0.5)
    >>> sr.R >= 0
    True
    """

    def __init__(self, martingale):
        """Wrap a martingale with the Shiryaev–Roberts statistic.

        Parameters
        ----------
        martingale : ConformalTestMartingale
            The underlying conformal test martingale whose increments drive the
            Shiryaev–Roberts statistic.
        """
        self.martingale = martingale
        self._n = 0
        self._sr_values = [0.0]  # R_0 = 0 (no terms in the sum)
        self._prev_logM = 0.0  # logM_{n-1}, starts at 0

    @property
    def R(self):
        """Current Shiryaev-Roberts statistic."""
        return self._sr_values[-1]

    @property
    def sr_values(self):
        """All Shiryaev-Roberts statistic values."""
        return self._sr_values

    def update(self, p: float) -> None:
        """Update the inner martingale and recompute the SR statistic.

        Uses the O(1) recursive formula (eq. 8.18 of ALRW2):
            R_n = (S_n / S_{n-1}) * (R_{n-1} + 1)
        """
        self.martingale.update(p)
        self._n += 1

        logM_n = self.martingale.logM
        # S_n / S_{n-1} = exp(logM_n - logM_{n-1})
        ratio = np.exp(logM_n - self._prev_logM)
        R_n = ratio * (self._sr_values[-1] + 1)
        self._sr_values.append(float(R_n))
        self._prev_logM = logM_n

    def alarm(self, threshold):
        """Check whether R_n exceeds the threshold.

        Parameters
        ----------
        threshold : float
            The alarm threshold.

        Returns
        -------
        bool
            True if R_n exceeds the threshold.
        """
        return self.R > threshold


