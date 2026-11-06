import numpy as np
import pytest

from online_cp.CPS import (
    DempsterHillConformalPredictiveSystem,
    KernelRidgePredictionMachine,
    MondrianPredictiveDistributionFunction,
    MondrianTreePredictionMachine,
    NearestNeighboursPredictionMachine,
    RidgePredictionMachine,
)
from online_cp.kernels import GaussianKernel
from online_cp.mondrian.tree import MondrianTree, _compute_subtree_counts


def _mondrian_category_labels(root, x, n, y):
    """Category labels via an independent path-based walk-up.

    Deliberately does not reuse ``_select_cps_category``: it re-derives the
    deepest node on the root→leaf(x) path holding at least one training
    point (indices ``< n``; the test point has index ``n``), using only the
    leaf indices and subtree counts.
    """

    def _iter_leaves(node):
        if node.is_leaf():
            yield node
        else:
            yield from _iter_leaves(node.left)
            yield from _iter_leaves(node.right)

    counts = _compute_subtree_counts(root)
    path, node = [root], root
    while not node.is_leaf():
        node = node.left if x[node.split_dim] <= node.split_loc else node.right
        path.append(node)
    for node in reversed(path):
        # Every node on the path also contains the test point (index n).
        if counts[id(node)] - 1 >= 1:
            idx = np.concatenate([leaf.indices for leaf in _iter_leaves(node)])
            return np.sort(y[np.asarray(idx)[np.asarray(idx) < n]])
    raise AssertionError("root contains no training data")


@pytest.fixture
def ridge_cps_dataset():
    rng = np.random.default_rng(2024)
    N = 100
    X = rng.normal(size=(N, 4))
    beta = np.array([2, 1, 0, 0])
    Y = X @ beta + rng.normal(scale=1, size=N)
    return X, Y


class TestRidgePredictionMachine:
    def test_cpd_monotone_in_tau(self, ridge_cps_dataset):
        """CPD(y, tau=0) <= CPD(y, tau=1) for all y."""
        X, Y = ridge_cps_dataset
        cps = RidgePredictionMachine(a=1, warnings=False)
        cps.learn_initial_training_set(X[:20], Y[:20])

        cpd = cps.predict_cpd(X[20])
        y_test = np.linspace(-5, 5, 50)
        for y in y_test:
            assert cpd(y, tau=0) <= cpd(y, tau=1) + 1e-12

    def test_cpd_bounded_zero_one(self, ridge_cps_dataset):
        """CPD values should be in [0, 1]."""
        X, Y = ridge_cps_dataset
        cps = RidgePredictionMachine(a=1, warnings=False)
        cps.learn_initial_training_set(X[:20], Y[:20])

        cpd = cps.predict_cpd(X[20])
        y_test = np.linspace(-10, 10, 100)
        for y in y_test:
            val = cpd(y, tau=0.5)
            assert -1e-10 <= val <= 1 + 1e-10

    def test_coverage(self, ridge_cps_dataset):
        """Prediction sets should have valid coverage."""
        X, Y = ridge_cps_dataset
        epsilon = 0.1
        cps = RidgePredictionMachine(a=1, warnings=False, epsilon=epsilon)
        cps.learn_initial_training_set(X[:20], Y[:20])

        rng = np.random.default_rng(42)
        covered = 0
        n_test = 60
        for i in range(20, 20 + n_test):
            tau = rng.uniform()
            cpd, precomputed = cps.predict_cpd(X[i], return_update=True)
            Gamma = cpd.predict_set(tau=tau, epsilon=epsilon)
            covered += int(Y[i] in Gamma)
            cps.learn_one(X[i], Y[i], precomputed=precomputed)

        coverage = covered / n_test
        assert coverage >= (1 - epsilon) - 0.10

    def test_learn_one_with_precomputed(self, ridge_cps_dataset):
        """Learning with precomputed should give same state as without."""
        X, Y = ridge_cps_dataset
        # With precomputed
        cps1 = RidgePredictionMachine(a=1, warnings=False)
        cps1.learn_initial_training_set(X[:20], Y[:20])
        _, precomputed = cps1.predict_cpd(X[20], return_update=True)
        cps1.learn_one(X[20], Y[20], precomputed=precomputed)

        # Without precomputed
        cps2 = RidgePredictionMachine(a=1, warnings=False)
        cps2.learn_initial_training_set(X[:20], Y[:20])
        cps2.learn_one(X[20], Y[20])

        assert np.allclose(cps1.XTXinv, cps2.XTXinv, atol=1e-10)
        assert np.allclose(cps1.y, cps2.y)


class TestNearestNeighboursPredictionMachine:
    def test_cpd_monotone_in_tau(self, ridge_cps_dataset):
        X, Y = ridge_cps_dataset
        cps = NearestNeighboursPredictionMachine(k=5)
        cps.learn_initial_training_set(X[:30], Y[:30])

        cpd = cps.predict_cpd(X[30])
        y_test = np.linspace(-5, 5, 30)
        for y in y_test:
            assert cpd(y, tau=0) <= cpd(y, tau=1) + 1e-12

    def test_coverage(self, ridge_cps_dataset):
        X, Y = ridge_cps_dataset
        epsilon = 0.1
        cps = NearestNeighboursPredictionMachine(k=5, epsilon=epsilon)
        cps.learn_initial_training_set(X[:30], Y[:30])

        rng = np.random.default_rng(42)
        covered = 0
        n_test = 40
        for i in range(30, 30 + n_test):
            tau = rng.uniform()
            cpd, precomputed = cps.predict_cpd(X[i], return_update=True)
            Gamma = cpd.predict_set(tau=tau, epsilon=epsilon)
            covered += int(Y[i] in Gamma)
            cps.learn_one(X[i], Y[i], precomputed=precomputed)

        coverage = covered / n_test
        assert coverage >= (1 - epsilon) - 0.10

    def test_repeated_labels(self):
        cps = NearestNeighboursPredictionMachine(k=3)
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([1.0, 1.0, 2.0, 2.0, 3.0])  # Duplicate labels
        cps.learn_initial_training_set(X, y)
        cpd = cps.predict_cpd(np.array([3.5]))
        # Should produce a valid CPD (monotone, bounded in [0, 1])
        pi0, pi1 = cpd(2.0)
        assert 0 <= pi0 <= pi1 <= 1


def _from_scratch_cpd(self, x):
    """Reference §7.5 CPD built from scratch (no cache).

    Builds every point's (k+1) nearest-neighbour set directly from the
    augmented distance matrix using the canonical (distance, global-index)
    tie-break (``np.lexsort``), then runs the book's §7.4 conformal sweep.
    This is the *intended* algorithm and is order-invariant; it is used as the
    ground truth for the cached semi-neighbour-refresh path. (The original
    dense-matrix implementation used ``np.argpartition`` + ``np.argsort``,
    which is unstable on distance ties and can leave a point's own index in its
    k-NN set when the point duplicates another.)
    """
    d = self.distance_func(self.X, x)
    D = self.update_distance_matrix(self.D, d)
    y = np.append(self.y, -np.inf)
    n = self.X.shape[0]
    k = self.k
    kk = min(k + 1, n)
    pos_keys = np.arange(n + 1)
    knn1 = np.empty((n + 1, k + 1), dtype=int)
    for i in range(n + 1):
        row = D[i, :].copy()
        row[i] = np.inf  # exclude self
        order = np.lexsort((pos_keys, row))
        knn1[i] = order[:kk]
    k_nearest = knn1[:, :k].T
    full, single, semi = set(), set(), set()
    knn_of_n = set(k_nearest.T[-1])
    for i, col in enumerate(k_nearest.T):
        in_n = i in knn_of_n
        in_i = n in col
        if in_n and in_i:
            full.add(i)
        elif in_n:
            single.add(i)
        elif in_i:
            semi.add(i)
    neighbours = full | single
    full_or_semi = full | semi
    idx_all = np.array(sorted(full | single | semi))
    Kprime = len(idx_all)
    Y = np.zeros(Kprime + 2)
    Y[0] = -np.inf
    Y[-1] = np.inf
    Y[1:-1] = y[idx_all]
    idx_mem = {i: idx_all[i - 1] for i in range(1, Kprime + 1)}
    sorted_indices = np.argsort(Y)[1:-1]
    Y.sort()
    Alpha = np.array([(y[k_nearest.T[i]] <= y_i).sum() for i, y_i in enumerate(y)])
    N = np.array([(Alpha == kk2).sum() for kk2 in range(k + 1)])
    L = -np.inf * np.ones(Kprime + 1)
    U = -np.inf * np.ones(Kprime + 1)
    L[0] = 0
    U[0] = N[0] / (n + 1)
    for kk2 in range(1, Kprime + 1):
        idx = idx_mem[sorted_indices[kk2 - 1]]
        if idx in neighbours:
            N[Alpha[-1]] -= 1
            Alpha[-1] += 1
            N[Alpha[-1]] += 1
        if idx in full_or_semi:
            N[Alpha[idx]] -= 1
            Alpha[idx] -= 1
            N[Alpha[idx]] += 1
        L[kk2] = N[: Alpha[-1]].sum() / (n + 1) if Alpha[-1] != 0 else 0
        U[kk2] = N[: Alpha[-1] + 1].sum() / (n + 1) if Alpha[-1] != 0 else N[0] / (n + 1)
    return L, U, Y


class TestSection75CPSpeedup:
    """§7.5 O(n) per-step speedup for the k-NN CPS (NearestNeighboursPredictionMachine).

    The cached per-point neighbour cache (test-free k-NN sets + the (k+1)-th
    neighbour distance threshold) refreshes only the *semi-neighbours* of each
    new example in ``learn_one`` and in ``predict_cpd``, rather than
    recomputing ``argpartition(D, k+1, axis=0)`` over the whole matrix. These
    tests pin the invariants: bit-identity with the from-scratch §7.5
    reference, order-invariance (permutation invariance), cache validity after
    streaming updates and save/load, and the warm-up regime.
    """

    @pytest.fixture
    def random_data(self):
        rng = np.random.default_rng(0)
        N = 120
        X = rng.normal(size=(N, 4))
        y = X @ rng.normal(size=4) + rng.normal(scale=0.5, size=N)
        return X, y

    @pytest.fixture
    def tie_data(self):
        """Rounded grid -> many distance ties (stress the tie-break)."""
        rng = np.random.default_rng(1)
        X = np.round(rng.normal(size=(100, 3)), 1)
        y = np.round(X @ np.array([1.0, 2.0, 3.0]), 0)
        return X, y

    @pytest.fixture
    def duplicate_data(self):
        """Integer grid -> duplicate points (stress the self-reference bug)."""
        rng = np.random.default_rng(2)
        X = np.round(rng.normal(size=(80, 2)), 0)
        y = X[:, 0] + X[:, 1]
        return X, y

    @pytest.mark.parametrize("k", [1, 3, 5])
    def test_bit_identity_random(self, random_data, k):
        """Cached predict_cpd is bit-identical to the from-scratch §7.5 CPD on
        generic (non-tie) data, over a streaming loop."""
        X, y = random_data
        n_init, n_test = 60, 40
        cps = NearestNeighboursPredictionMachine(k=k)
        cps.learn_initial_training_set(X[:n_init], y[:n_init])
        for t in range(n_test):
            x_test = X[n_init + t]
            L_c, U_c, Y_c = _from_scratch_cpd(cps, x_test)
            cpd = cps.predict_cpd(x_test)
            assert np.array_equal(cpd.L, L_c)
            assert np.array_equal(cpd.U, U_c)
            assert np.array_equal(cpd.Y, Y_c)
            cps.learn_one(x_test, y[n_init + t])

    def test_bit_identity_ties(self, tie_data):
        """Bit-identical on tie-rich data (the §7.5 tie-break must match the
        canonical (distance, index) selection)."""
        X, y = tie_data
        k, n_init, n_test = 2, 50, 30
        cps = NearestNeighboursPredictionMachine(k=k)
        cps.learn_initial_training_set(X[:n_init], y[:n_init])
        for t in range(n_test):
            x_test = X[n_init + t]
            L_c, U_c, Y_c = _from_scratch_cpd(cps, x_test)
            cpd = cps.predict_cpd(x_test)
            assert np.array_equal(cpd.L, L_c)
            assert np.array_equal(cpd.U, U_c)
            assert np.array_equal(cpd.Y, Y_c)
            cps.learn_one(x_test, y[n_init + t])

    def test_bit_identity_duplicates(self, duplicate_data):
        """Bit-identical on duplicate-point data. The original dense-matrix
        ``argpartition`` path had a latent self-reference bug here (a point
        could include itself in its own k-NN set when it duplicates another);
        the §7.5 cached path masks the diagonal and is therefore the
        mathematically correct selection.
        """
        X, y = duplicate_data
        k, n_init, n_test = 3, 40, 25
        cps = NearestNeighboursPredictionMachine(k=k)
        cps.learn_initial_training_set(X[:n_init], y[:n_init])
        for t in range(n_test):
            x_test = X[n_init + t]
            L_c, U_c, Y_c = _from_scratch_cpd(cps, x_test)
            cpd = cps.predict_cpd(x_test)
            assert np.array_equal(cpd.L, L_c)
            assert np.array_equal(cpd.U, U_c)
            assert np.array_equal(cpd.Y, Y_c)
            cps.learn_one(x_test, y[n_init + t])

    def test_warmup_regime(self):
        """n < k+1: every point is a semi-neighbour; the CPD must still be valid
        and bit-identical to the from-scratch reference."""
        rng = np.random.default_rng(3)
        N = 30
        X = rng.normal(size=(N, 2))
        y = rng.normal(size=N)
        k, n_init, n_test = 5, 8, 6
        cps = NearestNeighboursPredictionMachine(k=k)
        cps.learn_initial_training_set(X[:n_init], y[:n_init])
        for t in range(n_test):
            x_test = X[n_init + t]
            L_c, U_c, Y_c = _from_scratch_cpd(cps, x_test)
            cpd = cps.predict_cpd(x_test)
            assert np.array_equal(cpd.L, L_c)
            assert np.array_equal(cpd.U, U_c)
            assert np.array_equal(cpd.Y, Y_c)
            cps.learn_one(x_test, y[n_init + t])

    def test_order_invariance(self, tie_data):
        """Permuting the training bag must not change the CPD (bag-function /
        order-invariance), verified on tie-rich data where the tie-break is
        stressed.
        """
        X, y = tie_data
        k, n_init, n_test = 2, 50, 5
        rng = np.random.default_rng(7)
        perm = rng.permutation(n_init)
        cps1 = NearestNeighboursPredictionMachine(k=k)
        cps1.learn_initial_training_set(X[:n_init], y[:n_init])
        cps2 = NearestNeighboursPredictionMachine(k=k)
        cps2.learn_initial_training_set(X[:n_init][perm], y[:n_init][perm])
        x_test = X[n_init]
        cpd1 = cps1.predict_cpd(x_test)
        cpd2 = cps2.predict_cpd(x_test)
        assert np.array_equal(cpd1.Y, cpd2.Y)
        assert np.allclose(cpd1.L, cpd2.L, equal_nan=True)
        assert np.allclose(cpd1.U, cpd2.U, equal_nan=True)
        for t in range(n_test):
            xi = X[n_init + t]
            yi = y[n_init + t]
            c1 = cps1.predict_cpd(xi)
            c2 = cps2.predict_cpd(xi)
            assert np.array_equal(c1.Y, c2.Y)
            assert np.allclose(c1.L, c2.L, equal_nan=True)
            cps1.learn_one(xi, yi)
            cps2.learn_one(xi, yi)

    def test_learn_one_no_precomputed(self, random_data):
        """The no-precomputed ``learn_one`` path (which drives ``_refresh_cache``)
        must keep the cache valid: predictions stay bit-identical to the
        from-scratch reference across a pure streaming loop (no precompute)."""
        X, y = random_data
        k, n_init, n_test = 3, 40, 20
        cps = NearestNeighboursPredictionMachine(k=k)
        cps.learn_initial_training_set(X[:n_init], y[:n_init])
        for t in range(n_test):
            x_test = X[n_init + t]
            L_c, U_c, Y_c = _from_scratch_cpd(cps, x_test)
            cpd = cps.predict_cpd(x_test)
            assert np.array_equal(cpd.L, L_c)
            assert np.array_equal(cpd.U, U_c)
            assert np.array_equal(cpd.Y, Y_c)
            # No precomputed: exercises the incremental semi-neighbour refresh.
            cps.learn_one(x_test, y[n_init + t])

    def test_cache_valid_after_precomputed_learn(self, random_data):
        """After a ``learn_one`` that consumes a precomputed distance matrix,
        the cache must be rebuilt/refreshed and predictions stay bit-identical."""
        X, y = random_data
        k, n_init, n_test = 3, 40, 15
        cps = NearestNeighboursPredictionMachine(k=k)
        cps.learn_initial_training_set(X[:n_init], y[:n_init])
        for t in range(n_test):
            x_test = X[n_init + t]
            L_c, U_c, Y_c = _from_scratch_cpd(cps, x_test)
            cpd, precomputed = cps.predict_cpd(x_test, return_update=True)
            assert np.array_equal(cpd.L, L_c)
            assert np.array_equal(cpd.U, U_c)
            assert np.array_equal(cpd.Y, Y_c)
            cps.learn_one(x_test, y[n_init + t], precomputed=precomputed)

    def test_save_load_rebuilds_cache(self, random_data, tmp_path):
        """The neighbour cache is derived state; after a save/load round-trip
        it must be (re)built lazily so predictions are unchanged."""
        X, y = random_data
        k, n_init, n_test = 3, 30, 8
        cps = NearestNeighboursPredictionMachine(k=k)
        cps.learn_initial_training_set(X[:n_init], y[:n_init])
        for t in range(n_test):
            x_test = X[n_init + t]
            L_c, U_c, Y_c = _from_scratch_cpd(cps, x_test)
            cpd = cps.predict_cpd(x_test)
            assert np.array_equal(cpd.L, L_c)
            assert np.array_equal(cpd.U, U_c)
            assert np.array_equal(cpd.Y, Y_c)
            cps.learn_one(x_test, y[n_init + t])
        path = str(tmp_path / "cps_755.pkl")
        cps.save(path)
        cps2 = NearestNeighboursPredictionMachine.load(path)
        x_test = X[n_init + n_test]
        cpd_orig = cps.predict_cpd(x_test)
        cpd_reloaded = cps2.predict_cpd(x_test)
        assert np.array_equal(cpd_orig.Y, cpd_reloaded.Y)
        assert np.allclose(cpd_orig.L, cpd_reloaded.L, equal_nan=True)
        assert np.allclose(cpd_orig.U, cpd_reloaded.U, equal_nan=True)
        L_c, U_c, Y_c = _from_scratch_cpd(cps2, x_test)
        assert np.array_equal(cpd_reloaded.L, L_c)
        assert np.array_equal(cpd_reloaded.U, U_c)
        assert np.array_equal(cpd_reloaded.Y, Y_c)

    def test_cpd_monotone_and_bounded(self, tie_data):
        """The §7.5 CPD must remain a valid step function: monotone non-decreasing
        in tau, bounded in [0, 1], on tie-rich data.
        """
        X, y = tie_data
        k, n_init, n_test = 2, 50, 10
        cps = NearestNeighboursPredictionMachine(k=k, epsilon=0.1)
        cps.learn_initial_training_set(X[:n_init], y[:n_init])
        for t in range(n_test):
            x_test = X[n_init + t]
            cpd = cps.predict_cpd(x_test)
            for v in np.linspace(-5, 5, 51):
                pi0, pi1 = cpd(v, tau=0), cpd(v, tau=1)
                assert 0.0 <= pi0 <= pi1 <= 1.0 + 1e-12
            cps.learn_one(x_test, y[n_init + t])


class TestDempsterHillConformalPredictiveSystem:
    def test_cpd_monotone_in_tau(self):
        rng = np.random.default_rng(0)
        Y = rng.normal(size=50)
        cps = DempsterHillConformalPredictiveSystem()
        cps.learn_initial_training_set(Y)

        cpd = cps.predict_cpd()
        y_test = np.linspace(-3, 3, 30)
        for y in y_test:
            assert cpd(y, tau=0) <= cpd(y, tau=1) + 1e-12

    def test_coverage(self):
        rng = np.random.default_rng(42)
        Y = rng.normal(size=100)
        epsilon = 0.1
        cps = DempsterHillConformalPredictiveSystem(epsilon=epsilon)
        cps.learn_initial_training_set(Y[:20])

        covered = 0
        n_test = 60
        for i in range(20, 20 + n_test):
            tau = rng.uniform()
            cpd = cps.predict_cpd()
            Gamma = cpd.predict_set(tau=tau, epsilon=epsilon)
            covered += int(Y[i] in Gamma)
            cps.learn_one(Y[i])

        coverage = covered / n_test
        assert coverage >= (1 - epsilon) - 0.10

    def test_learn_one(self):
        cps = DempsterHillConformalPredictiveSystem()
        cps.learn_initial_training_set(np.array([1.0, 2.0, 3.0]))
        cps.learn_one(4.0)
        assert len(cps.y) == 4
        assert cps.y[-1] == 4.0


@pytest.fixture
def kernel_cps_dataset():
    rng = np.random.default_rng(2024)
    N = 80
    X = rng.normal(size=(N, 3))
    beta = np.array([2, 1, 0.5])
    Y = X @ beta + rng.normal(scale=1, size=N)
    return X, Y


class TestKernelRidgePredictionMachine:
    def test_cpd_monotone_in_tau(self, kernel_cps_dataset):
        """CPD(y, tau=0) <= CPD(y, tau=1) for all y."""
        X, Y = kernel_cps_dataset
        kernel = GaussianKernel(sigma=1.0)
        cps = KernelRidgePredictionMachine(kernel=kernel, a=1)
        cps.learn_initial_training_set(X[:15], Y[:15])

        cpd = cps.predict_cpd(X[15])
        y_test = np.linspace(-5, 5, 50)
        for y in y_test:
            assert cpd(y, tau=0) <= cpd(y, tau=1) + 1e-12

    def test_cpd_bounded_zero_one(self, kernel_cps_dataset):
        """CPD values should be in [0, 1]."""
        X, Y = kernel_cps_dataset
        kernel = GaussianKernel(sigma=1.0)
        cps = KernelRidgePredictionMachine(kernel=kernel, a=1)
        cps.learn_initial_training_set(X[:15], Y[:15])

        cpd = cps.predict_cpd(X[15])
        y_test = np.linspace(-10, 10, 100)
        for y in y_test:
            val = cpd(y, tau=0.5)
            assert -1e-10 <= val <= 1 + 1e-10

    def test_coverage(self, kernel_cps_dataset):
        """Prediction sets should have valid coverage."""
        X, Y = kernel_cps_dataset
        epsilon = 0.1
        kernel = GaussianKernel(sigma=1.0)
        cps = KernelRidgePredictionMachine(kernel=kernel, a=1, epsilon=epsilon)
        cps.learn_initial_training_set(X[:15], Y[:15])

        rng = np.random.default_rng(42)
        covered = 0
        n_test = 40
        for i in range(15, 15 + n_test):
            tau = rng.uniform()
            cpd, precomputed = cps.predict_cpd(X[i], return_update=True)
            Gamma = cpd.predict_set(tau=tau, epsilon=epsilon)
            covered += int(Y[i] in Gamma)
            cps.learn_one(X[i], Y[i], precomputed=precomputed)

        coverage = covered / n_test
        assert coverage >= (1 - epsilon) - 0.15

    def test_learn_one_with_precomputed(self, kernel_cps_dataset):
        """Learning with precomputed should give same state as without."""
        X, Y = kernel_cps_dataset
        kernel = GaussianKernel(sigma=1.0)

        # With precomputed
        cps1 = KernelRidgePredictionMachine(kernel=kernel, a=1)
        cps1.learn_initial_training_set(X[:15], Y[:15])
        _, precomputed = cps1.predict_cpd(X[15], return_update=True)
        cps1.learn_one(X[15], Y[15], precomputed=precomputed)

        # Without precomputed
        cps2 = KernelRidgePredictionMachine(kernel=kernel, a=1)
        cps2.learn_initial_training_set(X[:15], Y[:15])
        cps2.learn_one(X[15], Y[15])

        assert np.allclose(cps1.h_diag, cps2.h_diag, atol=1e-10)
        assert np.allclose(cps1.Hy, cps2.Hy, atol=1e-10)
        assert np.allclose(cps1.Kinv, cps2.Kinv, atol=1e-10)
        assert np.allclose(cps1.y, cps2.y)

    def test_learn_one_from_empty(self):
        """Can learn points one by one starting from empty."""
        kernel = GaussianKernel(sigma=1.0)
        cps = KernelRidgePredictionMachine(kernel=kernel, a=1)

        rng = np.random.default_rng(0)
        for _i in range(5):
            x = rng.normal(size=(1, 2))
            y = float(rng.normal())
            cps.learn_one(x, y)

        assert cps.X.shape == (5, 2)
        assert len(cps.y) == 5
        assert len(cps.h_diag) == 5
        assert len(cps.Hy) == 5

    def test_predict_after_learn_from_empty(self):
        """Can predict after learning from empty."""
        kernel = GaussianKernel(sigma=1.0)
        cps = KernelRidgePredictionMachine(kernel=kernel, a=1)

        rng = np.random.default_rng(1)
        for _i in range(10):
            x = rng.normal(size=(1, 2))
            y = float(rng.normal())
            cps.learn_one(x, y)

        x_test = rng.normal(size=(1, 2))
        cpd = cps.predict_cpd(x_test)
        # CPD should be valid
        assert 0 <= cpd(0.0, tau=0.5) <= 1


class TestMondrianTreePredictionMachine:
    def test_cpd_monotone_in_tau(self):
        """CPD(y, tau=0) <= CPD(y, tau=1) for all y."""
        rng = np.random.default_rng(2024)
        N = 50
        X = rng.normal(size=(N, 4))
        y = X.sum(axis=1) + rng.normal(scale=0.5, size=N)

        cps = MondrianTreePredictionMachine(lifetime=1.0)
        cps.learn_initial_training_set(X[:30], y[:30])

        cpd = cps.predict_cpd(X[30])
        y_test = np.linspace(y.min() - 2, y.max() + 2, 50)
        for y_val in y_test:
            assert cpd(y_val, tau=0) <= cpd(y_val, tau=1) + 1e-12

    def test_cpd_bounded_zero_one(self):
        """CPD values should be in [0, 1]."""
        rng = np.random.default_rng(2024)
        N = 50
        X = rng.normal(size=(N, 4))
        y = X.sum(axis=1) + rng.normal(scale=0.5, size=N)

        cps = MondrianTreePredictionMachine(lifetime=1.0)
        cps.learn_initial_training_set(X[:30], y[:30])

        cpd = cps.predict_cpd(X[30])
        y_test = np.linspace(y.min() - 5, y.max() + 5, 100)
        for y_val in y_test:
            val = cpd(y_val, tau=0.5)
            assert -1e-10 <= val <= 1 + 1e-10

    def test_coverage(self):
        """Prediction sets should have valid coverage.

        Uses pure-noise labels (independent of x) so that, given the
        partition, the test label is exactly exchangeable with the
        category labels — the regime where the leaf-local CPD is
        strictly valid.  A 1-D feature keeps the leaves large enough
        that the finite-sample deficit (≈ 2/(A+1)) stays small.  The
        threshold is deliberately well below 1 − ε; the sharp
        statistical check lives in the slow lane.
        """
        rng = np.random.default_rng(42)
        N_train, N_test = 200, 100
        X = rng.normal(size=(N_train + N_test, 1))
        y = rng.normal(size=N_train + N_test)

        epsilon = 0.1
        cps = MondrianTreePredictionMachine(lifetime=1.0, epsilon=epsilon)
        cps.learn_initial_training_set(X[:N_train], y[:N_train])

        covered = 0
        for i in range(N_train, N_train + N_test):
            tau = rng.uniform()
            cpd, precomputed = cps.predict_cpd(X[i], return_update=True)
            Gamma = cpd.predict_set(tau=tau, epsilon=epsilon)
            covered += int(y[i] in Gamma)
            cps.learn_one(X[i], y[i], precomputed=precomputed)

        coverage = covered / N_test
        assert coverage >= 0.75

    def test_learn_one(self):
        """learn_one should update the model correctly."""
        rng = np.random.default_rng(0)
        cps = MondrianTreePredictionMachine(lifetime=1.0)

        # Start with empty model (use learn_one after learn_initial_training_set)
        X = rng.normal(size=(5, 2))
        y = X.sum(axis=1)

        cps.learn_initial_training_set(X[:3], y[:3])
        assert cps.X.shape == (3, 2)
        assert len(cps.y) == 3

        cps.learn_one(X[3], y[3])
        assert cps.X.shape == (4, 2)
        assert len(cps.y) == 4

        cps.learn_one(X[4], y[4])
        assert cps.X.shape == (5, 2)
        assert len(cps.y) == 5

    def test_empty_training_set(self):
        """Should handle empty training set gracefully."""
        cps = MondrianTreePredictionMachine(lifetime=1.0)

        # Try to predict with empty training set
        cpd = cps.predict_cpd(np.array([0.0, 0.0]))
        # Should return a CPD that's valid (flat distribution)
        pi0, pi1 = cpd(0.0)
        assert 0.0 <= pi0 <= pi1 <= 1.0

    # -- Leaf-local CPD: category-structure verification ----------------------

    def test_cpd_knots_match_category_labels(self):
        """CPD knots must be exactly the training labels of the category.

        Verifies the leaf-local CPD against an independent re-growth of the
        partition (same seed) with its own descent and walk-up.
        """
        rng = np.random.default_rng(7)
        N, d = 40, 2
        X = rng.normal(size=(N, d))
        y = X.sum(axis=1) + rng.normal(scale=0.3, size=N)

        saw_proper_category = False
        for _ in range(30):
            x = rng.normal(size=d)
            cps = MondrianTreePredictionMachine(lifetime=1.0, rnd_state=123)
            cps.learn_initial_training_set(X, y)
            cpd = cps.predict_cpd(x)

            # Re-grow the same partition from the same RNG state.
            cps_ref = MondrianTreePredictionMachine(lifetime=1.0, rnd_state=123)
            cps_ref.learn_initial_training_set(X, y)
            root = MondrianTree.grow(
                np.vstack([X, x]), cps_ref.rnd_gen, lifetime=1.0, x_test=x
            ).root
            ref = _mondrian_category_labels(root, x, N, y)
            if ref.size < N:
                saw_proper_category = True
            np.testing.assert_array_equal(np.asarray(cpd.Y[1:-1]), ref)

        assert saw_proper_category, "no non-root category was exercised"

    def test_cpd_leaf_local(self):
        """An in-data test point must receive a leaf-local CPD.

        With 2 training points the leaf-local CPD carries at most the
        labels of one leaf — never the whole bag.
        """
        X = np.array([[0.0], [1.0]])
        y = np.array([10.0, 20.0])
        cps = MondrianTreePredictionMachine(lifetime=10.0, rnd_state=0)
        cps.learn_initial_training_set(X, y)
        cpd = cps.predict_cpd(np.array([0.0]))
        knots = np.asarray(cpd.Y[1:-1])
        assert knots.size == 1
        np.testing.assert_array_equal(knots, [10.0])

    def test_empty_leaf_walks_up(self):
        """An empty leaf must be replaced by the deepest ancestor with data.

        Test point 100 is split off from the data bag [0, 1] by the root
        split (which falls in (1, 100)), so its leaf holds no training
        data and the walk-up selects the root: both labels appear.
        """
        X = np.array([[0.0], [1.0]])
        y = np.array([10.0, 20.0])
        cps = MondrianTreePredictionMachine(lifetime=10.0, rnd_state=0)
        cps.learn_initial_training_set(X, y)
        cpd = cps.predict_cpd(np.array([100.0]))
        np.testing.assert_array_equal(np.asarray(cpd.Y[1:-1]), [10.0, 20.0])
        assert cpd(-50.0) == (0.0, 0.0)
        assert cpd(15.0) == (0.5, 0.5)  # between the two knots
        assert cpd(50.0) == (1.0, 1.0)


class TestMondrianPredictiveDistributionFunction:
    """Regression tests for the Mondrian CPD evaluation machinery."""

    @staticmethod
    def _cpd(labels):
        """Build a Mondrian CPD from category labels, as predict_cpd does."""
        A = len(labels)
        Y = np.empty(A + 2)
        Y[0] = -np.inf
        Y[-1] = np.inf
        Y[1:-1] = np.sort(labels)
        L = np.empty(A + 2)
        U = np.empty(A + 2)
        L[0] = U[0] = 0.0
        L[-1] = U[-1] = 1.0
        L[1:-1] = np.searchsorted(Y[1:-1], Y[1:-1], side="left") / A
        U[1:-1] = np.searchsorted(Y[1:-1], Y[1:-1], side="right") / A
        return MondrianPredictiveDistributionFunction(Y, L, U, epsilon=0.1)

    def test_bands_collapse_between_knots(self):
        """Away from knots the band must collapse to the single CDF value."""
        cpd = self._cpd([1.0, 2.0])
        assert cpd(-3.0) == (0.0, 0.0)
        assert cpd(0.5) == (0.0, 0.0)
        assert cpd(1.5) == (0.5, 0.5)
        assert cpd(3.0) == (1.0, 1.0)

    def test_bands_at_knots(self):
        """At a knot the band is [#{< y}/A, #{<= y}/A]."""
        cpd = self._cpd([1.0, 1.0, 2.0])
        assert cpd(1.0) == (0.0, 2.0 / 3.0)
        assert cpd(2.0) == (2.0 / 3.0, 1.0)

    def test_quantiles(self):
        """Quantiles must match the hand-computed smoothed CDF.

        Labels {1, 2}: smoothed CDF (tau=0.5) is 0.25 at the knot 1,
        0.5 on (1, 2), and 0.75 at the knot 2.
        """
        cpd = self._cpd([1.0, 2.0])
        assert cpd.quantile(0.25, 0.5) == 1.0
        assert np.isclose(cpd.quantile(0.5, 0.5), np.nextafter(1.0, np.inf))
        assert cpd.quantile(0.75, 0.5) == 2.0
        ps = cpd.predict_set(tau=0.5, epsilon=0.2)
        assert np.isfinite(ps.upper)

    def test_quantile_single_label(self):
        """A one-label CPD: smoothed CDF is tau at the knot, 1 above it."""
        cpd = self._cpd([5.0])
        # tau=0.5: knot value 0.5 -> q(0.5) is the knot itself
        assert cpd.quantile(0.5, 0.5) == 5.0
        # tau=0: knot value 0 -> q(0.3) just above the knot
        assert np.isclose(cpd.quantile(0.3, 0.0), np.nextafter(5.0, np.inf))
        # tau=1: knot value 1 -> q(0.7) is the knot
        assert cpd.quantile(0.7, 1.0) == 5.0

    def test_empty_category(self):
        """No labels: band [0, 1] everywhere, quantiles ±inf, set = R."""
        Y = np.array([-np.inf, np.inf])
        cpd = MondrianPredictiveDistributionFunction(
            Y, np.array([0.0, 1.0]), np.array([0.0, 1.0]), epsilon=0.1
        )
        assert cpd(0.0) == (0.0, 1.0)
        assert cpd.quantile(0.3, 0.5) == -np.inf
        assert cpd.quantile(0.7, 0.3) == np.inf
        ps = cpd.predict_set(tau=0.5, epsilon=0.2)
        assert ps.lower == -np.inf and ps.upper == np.inf


class TestMultiLevelCPD:
    def test_predict_set_multi_level(self):
        """CPD.predict_set with list of epsilons returns MultiLevelPredictionSet."""
        from online_cp.regressors import MultiLevelPredictionSet

        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 2))
        y = X @ np.array([1.0, -0.5]) + rng.normal(0, 0.3, 50)

        cps = RidgePredictionMachine(a=1.0)
        cps.learn_initial_training_set(X[:40], y[:40])

        cpd = cps.predict_cpd(X[40])
        tau = 0.5

        epsilons = [0.01, 0.05, 0.1, 0.2]
        result = cpd.predict_set(tau, epsilon=epsilons)

        assert isinstance(result, MultiLevelPredictionSet)
        assert result.levels == sorted(epsilons)
        assert len(result) == 4

        # Smaller epsilon => wider interval
        for i in range(len(epsilons) - 1):
            assert result[epsilons[i]].width() >= result[epsilons[i + 1]].width()


class TestHPDExactSweep:
    """Test predict_set(minimise_width=True) with exact O(n) sweep."""

    def test_hpd_narrower_or_equal(self, ridge_cps_dataset):
        """HPD interval should be ≤ equal-tailed interval in width."""
        X, Y = ridge_cps_dataset
        cps = RidgePredictionMachine(a=1.0, warnings=False)
        cps.learn_initial_training_set(X[:40], Y[:40])
        cpd = cps.predict_cpd(X[40])
        tau = 0.5
        epsilon = 0.1

        equal_tailed = cpd.predict_set(tau, epsilon=epsilon, minimise_width=False)
        hpd = cpd.predict_set(tau, epsilon=epsilon, minimise_width=True)

        assert hpd.width() <= equal_tailed.width() + 1e-10

    def test_hpd_contains_median(self, ridge_cps_dataset):
        """HPD interval should generally contain the median (for unimodal CPDs)."""
        X, Y = ridge_cps_dataset
        cps = RidgePredictionMachine(a=1.0, warnings=False)
        cps.learn_initial_training_set(X[:40], Y[:40])
        cpd = cps.predict_cpd(X[40])
        tau = 0.5
        epsilon = 0.1

        hpd = cpd.predict_set(tau, epsilon=epsilon, minimise_width=True)
        median = cpd.median(tau)
        assert hpd.lower <= median <= hpd.upper

    def test_hpd_width_monotone_in_epsilon(self, ridge_cps_dataset):
        """Smaller epsilon (higher coverage) => wider HPD."""
        X, Y = ridge_cps_dataset
        cps = RidgePredictionMachine(a=1.0, warnings=False)
        cps.learn_initial_training_set(X[:40], Y[:40])
        cpd = cps.predict_cpd(X[40])
        tau = 0.5

        widths = []
        for eps in [0.05, 0.1, 0.2, 0.3]:
            hpd = cpd.predict_set(tau, epsilon=eps, minimise_width=True)
            widths.append(hpd.width())
        # widths should be non-increasing as epsilon increases
        for i in range(len(widths) - 1):
            assert widths[i] >= widths[i + 1] - 1e-10


class TestMedian:
    def test_median_equals_quantile_half(self, ridge_cps_dataset):
        X, Y = ridge_cps_dataset
        cps = RidgePredictionMachine(a=1.0, warnings=False)
        cps.learn_initial_training_set(X[:40], Y[:40])
        cpd = cps.predict_cpd(X[40])
        tau = 0.5
        assert cpd.median(tau) == cpd.quantile(0.5, tau)
