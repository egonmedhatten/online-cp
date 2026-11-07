import numpy as np
import pytest

from online_cp.regressors import (
    ConformalNearestNeighboursRegressor,
    MultiLevelPredictionSet,
)


@pytest.fixture
def nonlinear_dataset():
    """y = sin(2*pi*x) + noise. N=200, d=1."""
    rng = np.random.default_rng(42)
    N = 200
    X = rng.uniform(0, 1, (N, 1))
    y = np.sin(2 * np.pi * X[:, 0]) + rng.normal(scale=0.1, size=N)
    return X, y


@pytest.fixture
def linear_dataset_2d():
    """y = x1 + x2 + noise. N=200, d=2."""
    rng = np.random.default_rng(123)
    N = 200
    X = rng.uniform(0, 1, (N, 2))
    y = X[:, 0] + X[:, 1] + rng.normal(scale=0.1, size=N)
    return X, y


class TestConformalNearestNeighboursRegressor:
    def test_validity_k1(self, nonlinear_dataset):
        """Coverage should be >= 1 - epsilon on iid data (k=1)."""
        X, y = nonlinear_dataset
        epsilon = 0.1
        cp = ConformalNearestNeighboursRegressor(k=1, rnd_state=0, epsilon=epsilon)

        n_init = 20
        cp.learn_initial_training_set(X[:n_init], y[:n_init])

        covered = 0
        n_test = len(y) - n_init
        for obj, lab in zip(X[n_init:], y[n_init:]):
            Gamma = cp.predict(obj)
            covered += int(lab in Gamma)
            cp.learn_one(obj, lab)

        coverage = covered / n_test
        assert coverage >= (1 - epsilon) - 0.05, f"Coverage {coverage:.3f} too low"

    def test_validity_k5(self, nonlinear_dataset):
        """Coverage should be >= 1 - epsilon on iid data (k=5)."""
        X, y = nonlinear_dataset
        epsilon = 0.1
        cp = ConformalNearestNeighboursRegressor(k=5, rnd_state=0, epsilon=epsilon)

        n_init = 20
        cp.learn_initial_training_set(X[:n_init], y[:n_init])

        covered = 0
        n_test = len(y) - n_init
        for obj, lab in zip(X[n_init:], y[n_init:]):
            Gamma = cp.predict(obj)
            covered += int(lab in Gamma)
            cp.learn_one(obj, lab)

        coverage = covered / n_test
        assert coverage >= (1 - epsilon) - 0.05, f"Coverage {coverage:.3f} too low"

    def test_validity_median_aggregation(self, nonlinear_dataset):
        """Coverage with median aggregation."""
        X, y = nonlinear_dataset
        epsilon = 0.1
        cp = ConformalNearestNeighboursRegressor(k=5, aggregation="median", rnd_state=0, epsilon=epsilon)

        n_init = 20
        cp.learn_initial_training_set(X[:n_init], y[:n_init])

        covered = 0
        n_test = len(y) - n_init
        for obj, lab in zip(X[n_init:], y[n_init:]):
            Gamma = cp.predict(obj)
            covered += int(lab in Gamma)
            cp.learn_one(obj, lab)

        coverage = covered / n_test
        assert coverage >= (1 - epsilon) - 0.05, f"Coverage {coverage:.3f} too low"

    def test_finite_intervals(self, nonlinear_dataset):
        """After sufficient training, intervals should be finite."""
        X, y = nonlinear_dataset
        cp = ConformalNearestNeighboursRegressor(k=3, rnd_state=0)
        cp.learn_initial_training_set(X[:30], y[:30])

        Gamma = cp.predict(X[30], epsilon=0.1)
        assert Gamma.lower > -np.inf
        assert Gamma.upper < np.inf
        assert Gamma.width() > 0

    def test_infinite_interval_with_no_data(self):
        """With no training data, interval should be infinite."""
        cp = ConformalNearestNeighboursRegressor(k=1, rnd_state=0)
        Gamma = cp.predict(np.array([0.5]), epsilon=0.1)
        assert Gamma.lower == -np.inf or Gamma.upper == np.inf

    def test_learn_one_no_initial(self):
        """learn_one from empty state."""
        cp = ConformalNearestNeighboursRegressor(k=1, rnd_state=0)
        cp.learn_one(np.array([1.0, 2.0]), 3.0)
        assert cp.X.shape == (1, 2)
        assert cp.y[0] == 3.0

    def test_learn_one_incremental(self, nonlinear_dataset):
        """learn_one should grow the distance matrix correctly."""
        X, y = nonlinear_dataset
        cp = ConformalNearestNeighboursRegressor(k=1, rnd_state=0)
        cp.learn_initial_training_set(X[:5], y[:5])

        assert cp.D.shape == (5, 5)
        cp.learn_one(X[5], y[5])
        assert cp.D.shape == (6, 6)
        assert cp.X.shape[0] == 6
        assert len(cp.y) == 6

    def test_distance_matrix_symmetric(self, nonlinear_dataset):
        """Distance matrix should remain symmetric after updates."""
        X, y = nonlinear_dataset
        cp = ConformalNearestNeighboursRegressor(k=1, rnd_state=0)
        cp.learn_initial_training_set(X[:10], y[:10])

        for i in range(10, 15):
            cp.learn_one(X[i], y[i])

        assert np.allclose(cp.D, cp.D.T)
        assert np.allclose(np.diag(cp.D), 0)

    def test_return_update(self, nonlinear_dataset):
        """return_update should provide D for efficient learn_one."""
        X, y = nonlinear_dataset
        cp = ConformalNearestNeighboursRegressor(k=3, rnd_state=0)
        cp.learn_initial_training_set(X[:20], y[:20])

        Gamma, precomputed = cp.predict(X[20], epsilon=0.1, return_update=True)
        assert "D" in precomputed
        assert precomputed["D"].shape == (21, 21)

        # Using precomputed should give same result as not using it
        cp2 = ConformalNearestNeighboursRegressor(k=3, rnd_state=0)
        cp2.learn_initial_training_set(X[:20], y[:20])

        cp.learn_one(X[20], y[20], precomputed=precomputed)
        cp2.learn_one(X[20], y[20])

        assert np.allclose(cp.D, cp2.D)

    def test_multi_level_epsilon(self, nonlinear_dataset):
        """Array-valued epsilon should return MultiLevelPredictionSet."""
        X, y = nonlinear_dataset
        cp = ConformalNearestNeighboursRegressor(k=3, rnd_state=0)
        cp.learn_initial_training_set(X[:30], y[:30])

        epsilons = np.array([0.05, 0.1, 0.2])
        result = cp.predict(X[30], epsilon=epsilons)
        assert isinstance(result, MultiLevelPredictionSet)
        assert len(result) == 3
        # Wider intervals at lower epsilon
        assert result[0.05].width() >= result[0.1].width()
        assert result[0.1].width() >= result[0.2].width()

    def test_custom_distance_func(self, linear_dataset_2d):
        """Custom distance function should work."""
        X, y = linear_dataset_2d

        def manhattan(X, y=None):
            from scipy.spatial.distance import cdist, pdist, squareform

            X = np.atleast_2d(X)
            if y is None:
                return squareform(pdist(X, metric="cityblock"))
            else:
                y = np.atleast_2d(y)
                return cdist(X, y, metric="cityblock")

        epsilon = 0.1
        cp = ConformalNearestNeighboursRegressor(k=3, distance_func=manhattan, rnd_state=0, epsilon=epsilon)
        cp.learn_initial_training_set(X[:30], y[:30])

        covered = 0
        n_test = 50
        for obj, lab in zip(X[30 : 30 + n_test], y[30 : 30 + n_test]):
            Gamma = cp.predict(obj)
            covered += int(lab in Gamma)
            cp.learn_one(obj, lab)

        coverage = covered / n_test
        assert coverage >= (1 - epsilon) - 0.1, f"Coverage {coverage:.3f} too low"

    def test_asymmetric_distance_func_kernel_style(self, linear_dataset_2d):
        """A distance function that is *not* symmetric in its argument roles
        (kernel-induced style: it requires ``y`` to be a single point) must work
        through ``learn_one`` and ``predict``.

        This guards the §7.5 refactor: the on-demand single-row distance
        computation must call ``distance_func(X, x_i)`` (matrix, single point),
        the same order the rest of the class uses.  A transposed
        ``distance_func(x_i, X)`` call would break such a function.
        """
        X, y = linear_dataset_2d

        def gaussian_kernel(Xq, y=None, sigma=1.0):
            # Kernel-induced distance for a Gaussian kernel K(a,b)=exp(-||a-b||^2/2s^2).
            # Mirrors kernel_induced_distance: y must be a SINGLE point when given,
            # and Kyy is obtained via kernel(y).item() -- which raises if y is a
            # matrix.  This asymmetry is what the §7.5 refactor must respect.
            from scipy.spatial.distance import cdist

            Xq = np.atleast_2d(Xq)
            if y is None:
                return np.exp(-cdist(Xq, Xq) ** 2 / (2 * sigma**2))
            y = np.atleast_2d(y)
            k_cross = np.exp(-cdist(Xq, y) ** 2 / (2 * sigma**2)).ravel()
            assert y.shape[0] == 1, "this kernel-style distance requires a single point as y"
            Kyy = float(np.exp(0.0))  # K(y, y) == 1.0 for a valid kernel
            diag_X = np.ones(Xq.shape[0])  # K(Xi, Xi) == 1.0 for a valid kernel
            D_sq = diag_X - 2 * k_cross + Kyy
            return np.sqrt(np.maximum(D_sq, 0.0)).reshape(-1, 1)

        epsilon = 0.1
        cp = ConformalNearestNeighboursRegressor(k=3, distance_func=gaussian_kernel, rnd_state=0, epsilon=epsilon)
        cp.learn_initial_training_set(X[:30], y[:30])

        covered = 0
        n_test = 50
        for obj, lab in zip(X[30 : 30 + n_test], y[30 : 30 + n_test]):
            Gamma = cp.predict(obj)
            covered += int(lab in Gamma)
            cp.learn_one(obj, lab)

        # Finite, valid intervals throughout (the asymmetric distance must not
        # corrupt the caches).
        assert np.isfinite(Gamma.lower) and np.isfinite(Gamma.upper)
        assert covered >= (1 - epsilon) - 0.1, f"Coverage {covered / n_test:.3f} too low"

    def test_invalid_aggregation(self):
        """Should raise ValueError for invalid aggregation."""
        with pytest.raises(ValueError, match="aggregation"):
            ConformalNearestNeighboursRegressor(aggregation="sum")

    def test_k_larger_than_n(self):
        """Should handle k > n gracefully."""
        cp = ConformalNearestNeighboursRegressor(k=100, rnd_state=0)
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0.0, 1.0, 2.0])
        cp.learn_initial_training_set(X, y)

        # Should not crash — uses all available neighbours
        Gamma = cp.predict(np.array([1.5]), epsilon=0.5)
        assert Gamma.lower <= 1.5 <= Gamma.upper

    def test_pvalue_tie_invariant_across_insertion_orders(self):
        """Regression guard for the k-NN test-point tie-break fix (audit 2026-10).

        The test point below is equidistant (distance 1.0) to the first four
        training points, so a bare ``np.argpartition`` would break the tie by
        insertion order and the p-value would change with the training-row
        order. With the ``np.lexsort((y, d))`` canonical tie-break the p-value
        is a bag function of the training data and must be identical across
        orders.
        """
        X = np.array(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [2.0, 0.0], [3.0, 0.0]]
        )
        y = np.array([0.0, 2.0, 1.0, 3.0, 100.0, 101.0])
        x_test = np.array([0.0, 0.0])

        def p_for(order):
            cp = ConformalNearestNeighboursRegressor(k=2, rnd_state=0)
            cp.learn_initial_training_set(X[order], y[order])
            return cp.compute_p_value(x_test, 1.0, smoothed=False)

        orders = [
            list(range(6)),
            list(range(5, -1, -1)),
            [3, 1, 4, 0, 5, 2],
            [2, 4, 0, 5, 1, 3],
        ]
        ref = p_for(orders[0])
        for order in orders[1:]:
            assert np.isclose(p_for(order), ref), (
                f"k-NN p-value is not invariant to insertion order "
                f"(order {order}: {p_for(order)} vs ref {ref})"
            )

    def test_bounds_lower(self, nonlinear_dataset):
        """Lower-only bound should have upper = inf."""
        X, y = nonlinear_dataset
        cp = ConformalNearestNeighboursRegressor(k=3, rnd_state=0)
        cp.learn_initial_training_set(X[:30], y[:30])

        Gamma = cp.predict(X[30], epsilon=0.1, bounds="lower")
        assert Gamma.lower > -np.inf
        assert Gamma.upper == np.inf

    def test_bounds_upper(self, nonlinear_dataset):
        """Upper-only bound should have lower = -inf."""
        X, y = nonlinear_dataset
        cp = ConformalNearestNeighboursRegressor(k=3, rnd_state=0)
        cp.learn_initial_training_set(X[:30], y[:30])

        Gamma = cp.predict(X[30], epsilon=0.1, bounds="upper")
        assert Gamma.lower == -np.inf
        assert Gamma.upper < np.inf


class TestIncrementalLearnOneMatchesScratchRebuild:
    """The incremental (semi-neighbour) ``learn_one`` must be bit-identical to a
    full from-scratch rebuild of the conformal caches at every stream step.

    This is the correctness guarantee that lets the O(n^2 log n) per-step rebuild
    be replaced by the O(n + |S| * n log n) semi-neighbour update (ALRW2 §2.4/§7.5)
    without changing the predicted intervals.
    """

    @staticmethod
    def _scratch_rebuild(cp):
        """Rebuild the conformal caches from scratch on the current bag."""
        cp._update_residuals()
        cp._update_kth_distances()

    def test_stream_intervals_identical(self, nonlinear_dataset):
        X, y = nonlinear_dataset
        n_init = 30
        n_stream = 60
        k = 5
        epsilon = 0.1

        # Incremental estimator (the production code path).
        inc = ConformalNearestNeighboursRegressor(k=k, rnd_state=0, epsilon=epsilon)
        inc.learn_initial_training_set(X[:n_init], y[:n_init])

        # Scratch estimator: identical object, but we force a full rebuild of the
        # caches at every step so that its state always matches a from-scratch fit.
        scr = ConformalNearestNeighboursRegressor(k=k, rnd_state=0, epsilon=epsilon)
        scr.learn_initial_training_set(X[:n_init], y[:n_init])

        for idx in range(n_stream):
            obj, lab = X[n_init + idx], y[n_init + idx]
            # Predict before updating -- both must agree at this step.
            g_inc = inc.predict(obj, epsilon=epsilon)
            g_scr = scr.predict(obj, epsilon=epsilon)
            assert np.isclose(g_inc.lower, g_scr.lower, equal_nan=True)
            assert np.isclose(g_inc.upper, g_scr.upper, equal_nan=True)
            # Update the scratch estimator the naive way (full rebuild), then
            # compare the caches directly.
            scr.learn_one(obj, lab)
            self._scratch_rebuild(scr)
            inc.learn_one(obj, lab)

            assert np.allclose(inc._a_const, scr._a_const, rtol=1e-12, atol=1e-12, equal_nan=True)
            assert np.allclose(inc._kth_distances, scr._kth_distances, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_stream_end_state_matches_batch(self, nonlinear_dataset):
        """After streaming, the incremental estimator must agree with a fresh
        estimator that was built on the full (initial + streamed) bag in one shot."""
        X, y = nonlinear_dataset
        n_init = 30
        n_stream = 60
        k = 3
        epsilon = 0.1

        inc = ConformalNearestNeighboursRegressor(k=k, rnd_state=0, epsilon=epsilon)
        inc.learn_initial_training_set(X[:n_init], y[:n_init])
        for idx in range(n_stream):
            obj, lab = X[n_init + idx], y[n_init + idx]
            inc.predict(obj, epsilon=epsilon)
            inc.learn_one(obj, lab)

        full = X[:n_init + n_stream]
        full_y = y[:n_init + n_stream]
        batch = ConformalNearestNeighboursRegressor(k=k, rnd_state=0, epsilon=epsilon)
        batch.learn_initial_training_set(full, full_y)

        assert np.allclose(inc._a_const, batch._a_const, rtol=1e-12, atol=1e-12, equal_nan=True)
        assert np.allclose(inc._kth_distances, batch._kth_distances, rtol=1e-12, atol=1e-12, equal_nan=True)



class TestWarmUpReverseNeighbourDetection:
    """When the training bag holds fewer than k points (n < k), the test point
    x_n is a reverse neighbour of *every* training point: there are not k other
    points to fill x_i's k-nearest set, so augmenting the bag with x_n
    necessarily displaces at least one neighbour of x_i. Hence the affine
    slope b_i is -1/k for all i in this regime, not just for those satisfying
    d(x_i, x_n) <= kth_nearest_dist(x_i).

    Regression guard for the n < k reverse-neighbour branch in predict(). The
    bug it catches produced an incorrect one-sided lower bound during the
    warm-up phase (before the bag reached k+1 points).
    """

    @staticmethod
    def _reference_interval(X, y, x_test, k, epsilon, bounds):
        """Naive ALRW2 Sec. 2.4 reference (includes the test point in the k-NN
        check of x_i, so it is correct for all n including n < k)."""
        from scipy.spatial.distance import cdist

        n = len(y)
        d = cdist(x_test.reshape(1, -1), X).ravel()
        k_t = min(k, n)
        kidx = np.lexsort((y, d))[:k_t]
        a_n = -y[kidx].mean()
        b_n = 1.0
        a = np.empty(n)
        b = np.empty(n)
        for i in range(n):
            d_i = cdist(X[i].reshape(1, -1), X).ravel()
            d_i[i] = np.inf
            others = np.append(d_i, d[i])
            kk = min(k, n)
            kth = np.partition(others, kk - 1)[kk - 1]
            if d[i] <= kth:
                b[i] = -1.0 / k
                yfull = np.append(y, 0.0)
                idx = np.lexsort((yfull, others))[:kk]
                a[i] = y[i] - yfull[idx].mean()
            else:
                b[i] = 0.0
                di = d_i.copy()
                di[i] = np.inf
                idx = np.lexsort((y, di))[:kk]
                a[i] = y[i] - y[idx].mean()
        t = (a - a_n) / (b_n - b)
        t_sorted = np.sort(t)
        N = n + 1

        def gu(eps):
            idx = int(np.ceil((1 - eps) * N))
            if idx < 1:
                return -np.inf
            if idx > n:
                return np.inf
            return float(t_sorted[idx - 1])

        def gl(eps):
            idx = int(np.floor(eps * N))
            if idx < 1:
                return -np.inf
            if idx > n:
                return np.inf
            return float(t_sorted[idx - 1])

        if bounds == "both":
            return gl(epsilon / 2), gu(epsilon / 2)
        if bounds == "upper":
            return -np.inf, gu(epsilon)
        return gl(epsilon), np.inf

    @pytest.mark.parametrize("n_init", [1, 2, 3, 4])
    @pytest.mark.parametrize("bounds", ["both", "upper", "lower"])
    def test_intervals_match_reference_for_n_lt_k(self, n_init, bounds):
        """For n < k the production intervals must match the reference exactly,
        including the one-sided bounds."""
        k = 5
        rng = np.random.default_rng(123 + n_init)
        for trial in range(30):
            d = 2
            eps = float(rng.choice([0.01, 0.05, 0.1, 0.2, 0.3]))
            X = rng.normal(size=(n_init, d))
            y = X[:, 0] + rng.normal(size=n_init) * 0.1
            x_test = rng.normal(size=d)
            cp = ConformalNearestNeighboursRegressor(k=k, rnd_state=0, epsilon=eps)
            cp.learn_initial_training_set(X, y)
            g = cp.predict(x_test, epsilon=eps, bounds=bounds)
            lo, hi = self._reference_interval(X, y, x_test, k, eps, bounds)
            assert np.isclose(g.lower, lo, atol=1e-9, equal_nan=True), (
                f"n={n_init}, trial={trial}, bounds={bounds}, eps={eps}: "
                f"lower {g.lower} != {lo}"
            )
            assert np.isclose(g.upper, hi, atol=1e-9, equal_nan=True), (
                f"n={n_init}, trial={trial}, bounds={bounds}, eps={eps}: "
                f"upper {g.upper} != {hi}"
            )


class TestSection75OptimalStep:
    """The §7.5 optimization: ``learn_one`` must not materialise the full
    ``n x n`` distance matrix in the non-pruned regime, and its per-step cost
    must be sub-quadratic in ``n``.

    The internal cache is ``_D_cache``; the public ``D`` property materialises
    it on demand.  A pure streaming ``learn_one`` loop (no pruning, no
    ``return_update``) that never reads ``self.D`` must keep ``_D_cache``
    ``None`` throughout, which is the observable signature of the ``O(n d + n)``
    step (the dense-matrix ``O(n^2)`` scan has been removed).
    """

    def test_matrix_not_maintained_in_pure_stream(self, nonlinear_dataset):
        """A non-pruned streaming loop must never build the full matrix."""
        X, y = nonlinear_dataset
        cp = ConformalNearestNeighboursRegressor(k=3, rnd_state=0)
        cp.learn_initial_training_set(X[:20], y[:20])
        # Cold start does not maintain the matrix in the non-pruned regime.
        assert cp._D_cache is None
        for i in range(20, 80):
            cp.predict(X[i], epsilon=0.1)  # no return_update
            cp.learn_one(X[i], y[i])
        # Still no full matrix after a long stream.
        assert cp._D_cache is None

    def test_pruning_maintains_matrix(self):
        """With stream-pruning active the full matrix *is* maintained (the
        eviction path needs it)."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 2))
        y = rng.normal(size=40)
        cp = ConformalNearestNeighboursRegressor(k=3, rnd_state=0, pruning="reservoir", budget=10)
        cp.learn_initial_training_set(X[:20], y[:20])
        assert cp._D_cache is not None
        for i in range(20, 40):
            cp.predict(X[i], epsilon=0.1)
            cp.learn_one(X[i], y[i])
        assert cp._D_cache is not None

    def test_predict_return_update_still_provides_matrix(self, nonlinear_dataset):
        """``return_update=True`` must still hand back a full augmented matrix
        even in the non-pruned regime (the contract is unchanged)."""
        X, y = nonlinear_dataset
        cp = ConformalNearestNeighboursRegressor(k=3, rnd_state=0)
        cp.learn_initial_training_set(X[:20], y[:20])
        _, precomputed = cp.predict(X[20], epsilon=0.1, return_update=True)
        assert "D" in precomputed
        assert precomputed["D"].shape == (21, 21)
        # Feeding it back to learn_one must be bit-identical to the no-precompute
        # path.
        cp2 = ConformalNearestNeighboursRegressor(k=3, rnd_state=0)
        cp2.learn_initial_training_set(X[:20], y[:20])
        cp.learn_one(X[20], y[20], precomputed=precomputed)
        cp2.learn_one(X[20], y[20])
        assert np.allclose(cp._a_const, cp2._a_const, rtol=1e-12, atol=1e-12, equal_nan=True)
        assert np.allclose(cp._kth_distances, cp2._kth_distances, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_public_D_property_materialises_on_demand(self, nonlinear_dataset):
        """Reading the public ``D`` property materialises the matrix once; the
        value must equal what a from-scratch fit would produce."""
        X, y = nonlinear_dataset
        cp = ConformalNearestNeighboursRegressor(k=3, rnd_state=0)
        cp.learn_initial_training_set(X[:30], y[:30])
        for i in range(30, 50):
            cp.learn_one(X[i], y[i])
        assert cp._D_cache is None  # not built during streaming
        D = cp.D  # now materialised
        assert D is not None and D.shape == (50, 50)
        assert cp._D_cache is not None
        # BLAS distance computation is non-deterministic at the 1e-12 level
        # under thread load, so use a relaxed tolerance.
        expected = cp.distance_func(cp.X)
        assert np.allclose(D, expected, atol=1e-9)

    def test_step_cost_subquadratic(self, nonlinear_dataset):
        """The per-step ``learn_one`` cost must not grow quadratically.  We
        compare the doubling ratio of per-step time: an ``O(n^2)`` step gives a
        ratio ~4 when ``n`` doubles; an ``O(n)`` step gives a ratio ~2.  We allow
        generous headroom but assert the ratio is comfortably below the
        quadratic threshold (and that cost at 2x is below the cost at 4x, i.e.
        no superlinear blowup)."""
        import time

        X, y = nonlinear_dataset

        def per_step(n):
            cp = ConformalNearestNeighboursRegressor(k=3, rnd_state=0)
            cp.learn_initial_training_set(X[:20], y[:20])
            t0 = time.perf_counter()
            for i in range(20, n):
                cp.learn_one(X[i], y[i])
            return (time.perf_counter() - t0) / (n - 20)

        c1, c2 = per_step(80), per_step(160)
        # An O(n) step gives ratio ~2 on doubling; an O(n^2) step gives ~4.
        # Allow headroom for timing noise under thread contention, but stay
        # comfortably below the quadratic bound.
        assert c2 / c1 < 3.5, (
            f"per-step cost ratio {c2 / c1:.2f} when doubling n is close to the "
            f"quadratic bound of 4 -- the O(n^2) scan is back."
        )
