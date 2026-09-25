"""Mondrian tree engine and the standalone :class:`MondrianTree` core.

This module owns a single sampled Mondrian partition and everything needed to
grow, traverse and visualise it. It is deliberately free of any dependency on
the conformal / Venn predictor modules so that those predictors can import the
engine without creating an import cycle.

The public entry point is :class:`MondrianTree`. Conformal predictors
(``ConformalMondrianTree*`` / ``ConformalMondrianForest*`` in
``online_cp.classifiers`` / ``online_cp.regressors``) and
:class:`~online_cp.venn.MondrianVennPredictor` are thin adapters that grow a
:class:`MondrianTree` and then apply their own nonconformity / taxonomy logic on
top of the leaf statistics.

Theory
------
The partition depends only on the bounding boxes of ``X`` (a bag function of the
feature values, independent of ordering), so a tree grown from the augmented bag
``{train ∪ (x_test, y_candidate)}`` yields an exact conformal predictor
(ALRW2 §2.2.9). The self-consistency property (Balog §2.3) guarantees that a tree
at lifetime ``λ`` is a deterministic truncation of the master tree grown with the
same RNG — this underpins the unsupervised ``lifetime`` resolvers.

References
----------
- Lakshminarayanan, Roy, Teh (2014). "Mondrian forests: Efficient online random
  forests." NeurIPS.
- Balog (2015). "The Mondrian Process for Machine Learning." (thesis)
- Vovk, Gammerman, Shafer (2022). "Algorithmic Learning in a Random World"
  (2nd ed.), §2.2.9, §4.6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import graphviz


# ---------------------------------------------------------------------------
# Mondrian tree engine (node structure, sampling, traversal, visualisation)
# ---------------------------------------------------------------------------

@dataclass
class _MondrianNode:
    """Node in a Mondrian tree.

    Attributes:
        split_dim: Feature dimension to split on (-1 for leaf nodes).
        split_loc: Split location along split_dim.
        split_time: Time of split (from exponential distribution).
        parent_time: Time of parent split (0 for root).
        left: Left child (feature value <= split_loc).
        right: Right child (feature value > split_loc).
        lower_bounds: Bounding box lower corner (d,).
        upper_bounds: Bounding box upper corner (d,).
        indices: Indices of training points in this subtree's leaf (None for internal nodes).
        counts: Class count vector for leaf (None for internal nodes).
    """

    split_dim: int
    split_loc: float
    split_time: float
    parent_time: float
    left: _MondrianNode | None = None
    right: _MondrianNode | None = None
    lower_bounds: NDArray = field(default_factory=np.array)
    upper_bounds: NDArray = field(default_factory=np.array)
    indices: NDArray | None = None
    counts: NDArray | None = None

    def is_leaf(self) -> bool:
        """Check if node is a leaf."""
        return self.split_dim == -1

    def n_points(self) -> int:
        """Number of points in this leaf (0 for internal nodes)."""
        if self.indices is None:
            return 0
        return len(self.indices)


def _sample_mondrian_tree(
    rng: np.random.Generator,
    X: NDArray,
    indices: NDArray,
    parent_time: float,
    lifetime: float,
    verbose: int = 0,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    feature_weights: NDArray | None = None,
    _depth: int = 0,
) -> _MondrianNode:
    """Recursively build a Mondrian tree from data.

    Algorithm (SampleMondrianBlock from Lakshminarayanan et al. 2014):

    1. Compute bounding box of X[indices]
    2. Sample E ~ Exp(rate = sum of ranges)
    3. If parent_time + E >= lifetime: return leaf
    4. Sample split dimension proportional to range sizes
    5. Sample split location uniformly within range
    6. Partition indices into left (X[i, dim] <= loc) and right
    7. Recurse for both children

    Args:
        rng: NumPy random generator
        X: Feature array (n, d)
        indices: Indices of points to include (n_sub,)
        parent_time: Time of parent split (0 for root)
        lifetime: Maximum total lifetime budget (time >= lifetime → leaf)
        verbose: Verbosity level
        max_depth: Hard cap on tree depth. A node at depth ``_depth >= max_depth``
            becomes a leaf regardless of remaining lifetime. ``None`` means no cap
            (default, identical behaviour to the original algorithm).
        min_samples_leaf: Minimum number of points allowed in each leaf.
            A split is refused if it would create a child with fewer than
            ``min_samples_leaf`` points.  Default 1 is a true identity: the
            check ``n_points < 2`` matches the existing ``n_points <= 1``
            single-point stop condition, and the post-partition check
            ``len(child) < 1`` matches the existing empty-side check.
        feature_weights: Optional per-dimension weights for anisotropic
            splitting.  When provided, the Poisson split rate for dimension d
            becomes ``feature_weights[d] * ranges[d]`` instead of
            ``ranges[d]``.  Dimensions with ``feature_weights[d] == 0`` are
            never split.  Must be non-negative with shape ``(d,)``.  ``None``
            (default) recovers the standard isotropic Mondrian behaviour.
        _depth: Internal recursion counter — do not pass from outside.

    Returns:
        Root node of built tree
    """
    n_points = len(indices)
    d = X.shape[1]

    # Compute bounding box
    X_subset = X[indices]
    lower_bounds = X_subset.min(axis=0)
    upper_bounds = X_subset.max(axis=0)
    ranges = upper_bounds - lower_bounds

    # Compute exponential rate (anisotropic weighting if feature_weights provided).
    # _weighted_ranges[d] = w_d * (u_d - l_d); split probability ∝ _weighted_ranges[d].
    if feature_weights is not None:
        _weighted_ranges = feature_weights * ranges
        rate = _weighted_ranges.sum()
    else:
        _weighted_ranges = ranges
        rate = _weighted_ranges.sum()

    # Check stopping conditions: no variance, single point, or too few for a split
    if n_points < 2 * min_samples_leaf or rate == 0:
        # Leaf node
        node = _MondrianNode(
            split_dim=-1,
            split_loc=np.nan,
            split_time=parent_time,  # Use parent time for leaves
            parent_time=parent_time,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            indices=indices.copy(),
            counts=None,  # Will be filled in later
        )
        return node

    # Hard depth cap: make this node a leaf if depth budget is exhausted
    if max_depth is not None and _depth >= max_depth:
        node = _MondrianNode(
            split_dim=-1,
            split_loc=np.nan,
            split_time=parent_time,
            parent_time=parent_time,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            indices=indices.copy(),
            counts=None,
        )
        return node

    # Sample E ~ Exp(rate)
    E = rng.exponential(scale=1.0 / rate) if rate > 0 else np.inf
    split_time = parent_time + E

    # Check if we've exhausted lifetime
    if split_time >= lifetime:
        # Leaf node
        node = _MondrianNode(
            split_dim=-1,
            split_loc=np.nan,
            split_time=split_time,
            parent_time=parent_time,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            indices=indices.copy(),
            counts=None,  # Will be filled in later
        )
        return node

    # Sample split dimension (proportional to weighted ranges; rate > 0 guaranteed here)
    split_probs = _weighted_ranges / rate
    split_dim = rng.choice(d, p=split_probs)

    # Sample split location uniformly in the range
    split_loc = rng.uniform(lower_bounds[split_dim], upper_bounds[split_dim])

    # Partition indices
    mask_left = X[indices, split_dim] <= split_loc
    indices_left = indices[mask_left]
    indices_right = indices[~mask_left]

    # Handle edge case or min_samples_leaf violation: treat as leaf
    if len(indices_left) < min_samples_leaf or len(indices_right) < min_samples_leaf:
        node = _MondrianNode(
            split_dim=-1,
            split_loc=np.nan,
            split_time=split_time,
            parent_time=parent_time,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            indices=indices.copy(),
            counts=None,
        )
        return node

    # Recurse for children
    left_child = _sample_mondrian_tree(
        rng, X, indices_left, split_time, lifetime, verbose=verbose,
        max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        feature_weights=feature_weights, _depth=_depth + 1,
    )
    right_child = _sample_mondrian_tree(
        rng, X, indices_right, split_time, lifetime, verbose=verbose,
        max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        feature_weights=feature_weights, _depth=_depth + 1,
    )

    # Internal node
    node = _MondrianNode(
        split_dim=split_dim,
        split_loc=split_loc,
        split_time=split_time,
        parent_time=parent_time,
        left=left_child,
        right=right_child,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        indices=None,
        counts=None,
    )

    return node


def _extend_mondrian(
    node: _MondrianNode,
    x: NDArray,
    x_idx: int,
    rng: np.random.Generator,
    lifetime: float,
    feature_weights: NDArray | None = None,
) -> _MondrianNode:
    """Project a new point ``x`` into an existing Mondrian tree (ExtendMondrianBlock).

    Returns the root of the *augmented* tree. This is a **functional** update: only
    the O(depth) nodes on the path to ``x`` are freshly allocated (via
    :func:`dataclasses.replace`); every off-path subtree is shared with *node*, and
    *node* itself is never mutated. Callers therefore get a new tree that either
    replaces the persistent one (``learn_one``) or can be scored and discarded
    (transductive ``predict``) without touching the original.

    The Mondrian process is projective (Roy & Teh 2009; Lakshminarayanan et al.
    2014, Algorithm 3): the tree returned here has the same *law* as
    :func:`_sample_mondrian_tree` grown in one batch on the augmented point set.
    Only the standard (uncapped) process is projective, so this routine assumes a
    fixed float ``lifetime``, no ``max_depth`` cap and ``min_samples_leaf == 1``.

    Args:
        node: Root of the tree to extend.
        x: New point, shape ``(d,)``.
        x_idx: Index assigned to ``x`` (its row in the caller's augmented ``X``).
        rng: Random generator (advanced in place).
        lifetime: The tree's lifetime budget.
        feature_weights: Optional per-dimension split weights (must match the tree).

    Returns:
        Root of the augmented tree.
    """
    lower = node.lower_bounds
    upper = node.upper_bounds
    e_lower = np.maximum(lower - x, 0.0)
    e_upper = np.maximum(x - upper, 0.0)
    extent = e_lower + e_upper
    weighted = feature_weights * extent if feature_weights is not None else extent
    rate = float(weighted.sum())

    exp_draw = rng.exponential(scale=1.0 / rate) if rate > 0 else np.inf
    # A leaf has no split of its own; its box "survives" until the lifetime budget.
    upper_time = node.split_time if not node.is_leaf() else lifetime

    if node.parent_time + exp_draw < upper_time:
        # Introduce a brand-new split *above* this node, carving x off from its box.
        split_dim = int(rng.choice(len(x), p=weighted / rate))
        if x[split_dim] > upper[split_dim]:
            split_loc = float(rng.uniform(upper[split_dim], x[split_dim]))
        else:
            split_loc = float(rng.uniform(x[split_dim], lower[split_dim]))
        new_time = node.parent_time + exp_draw

        new_leaf = _MondrianNode(
            split_dim=-1,
            split_loc=np.nan,
            split_time=new_time,
            parent_time=new_time,
            lower_bounds=x.copy(),
            upper_bounds=x.copy(),
            indices=np.array([x_idx]),
            counts=None,
        )
        # The existing subtree is reborn at new_time; it keeps its original box
        # (x lies outside it, in the sibling leaf).
        existing = replace(node, parent_time=new_time)
        if x[split_dim] <= split_loc:
            left, right = new_leaf, existing
        else:
            left, right = existing, new_leaf

        return _MondrianNode(
            split_dim=split_dim,
            split_loc=split_loc,
            split_time=new_time,
            parent_time=node.parent_time,
            left=left,
            right=right,
            lower_bounds=np.minimum(lower, x),
            upper_bounds=np.maximum(upper, x),
            indices=None,
            counts=None,
        )

    # No new split here: grow this node's box to include x and descend.
    new_lower = np.minimum(lower, x)
    new_upper = np.maximum(upper, x)
    if node.is_leaf():
        return replace(
            node,
            lower_bounds=new_lower,
            upper_bounds=new_upper,
            indices=np.append(node.indices, x_idx),
        )
    if x[node.split_dim] <= node.split_loc:
        new_left = _extend_mondrian(node.left, x, x_idx, rng, lifetime, feature_weights)
        return replace(node, lower_bounds=new_lower, upper_bounds=new_upper, left=new_left)
    new_right = _extend_mondrian(node.right, x, x_idx, rng, lifetime, feature_weights)
    return replace(node, lower_bounds=new_lower, upper_bounds=new_upper, right=new_right)


def _find_leaf(node: _MondrianNode, x: NDArray) -> _MondrianNode:
    """Traverse tree to find leaf containing x.

    Args:
        node: Root node
        x: Feature vector (d,)

    Returns:
        Leaf node containing x
    """
    current = node
    while not current.is_leaf():
        if x[current.split_dim] <= current.split_loc:
            current = current.left
        else:
            current = current.right
    return current


def _collect_leaves(node: _MondrianNode) -> list[_MondrianNode]:
    """Collect all leaf nodes in tree.

    Args:
        node: Root node

    Returns:
        List of all leaf nodes
    """
    if node.is_leaf():
        return [node]
    leaves = []
    leaves.extend(_collect_leaves(node.left))
    leaves.extend(_collect_leaves(node.right))
    return leaves


def _iter_nodes(node: _MondrianNode, _depth: int = 0, _parent_id: str | None = None) -> list[tuple]:
    """Depth-first traversal of all nodes in tree.

    Yields (node, depth, parent_id, node_id) tuples in pre-order.

    Args:
        node: Root node
        _depth: Current depth (internal)
        _parent_id: Parent node ID (internal)

    Yields:
        (node, depth, parent_id, node_id) tuples
    """
    node_id = f"node_{id(node)}"
    yield (node, _depth, _parent_id, node_id)
    if not node.is_leaf():
        yield from _iter_nodes(node.left, _depth + 1, node_id)
        yield from _iter_nodes(node.right, _depth + 1, node_id)


def _tree_struct_stats(node: _MondrianNode) -> dict:
    """Compute structural statistics of a tree.

    Args:
        node: Root node

    Returns:
        Dict with keys: n_nodes, n_leaves, n_branches, height
    """
    stats = {"n_nodes": 0, "n_leaves": 0, "n_branches": 0, "height": 0}

    def traverse(n: _MondrianNode, depth: int) -> int:
        stats["n_nodes"] += 1
        stats["height"] = max(stats["height"], depth)
        if n.is_leaf():
            stats["n_leaves"] += 1
        else:
            stats["n_branches"] += 1
            traverse(n.left, depth + 1)
            traverse(n.right, depth + 1)

    traverse(node, 0)
    return stats


def _assign_counts(
    root: _MondrianNode,
    y: NDArray,
    label_to_idx: dict,
    K: int,
    n_train: int | None = None,
) -> None:
    """Assign class count vectors to all leaf nodes.

    Modifies the tree in-place by setting counts for each leaf based on
    the true labels of points in that leaf.

    Args:
        root: Root node of tree
        y: Class labels — any hashable values.
        label_to_idx: Mapping from label to integer index 0..K-1
        K: Number of classes
        n_train: If given, skip indices >= n_train (the test point's structural
                 slot in X_aug), producing training-only base counts.
    """
    leaves = _collect_leaves(root)
    for leaf in leaves:
        counts = np.zeros(K, dtype=np.int64)
        for idx in leaf.indices:
            if n_train is not None and idx >= n_train:
                continue  # skip test point's structural slot
            label_idx = label_to_idx[y[idx]]
            counts[label_idx] += 1
        leaf.counts = counts


# ---------------------------------------------------------------------------
# Visualisation helpers (matplotlib, lazily imported)
# ---------------------------------------------------------------------------

def _get_ax(ax):
    """Return ax if given, else create a new figure and return its Axes."""
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
    return ax


def _node_x_positions(node: _MondrianNode, max_depth: int | None = None, _depth: int = 0):
    """Recursively compute x-positions for tree layout.

    Leaves are numbered left-to-right; internal-node x is the mean of its
    children's x.  Returns a dict mapping id(node) -> float x-position.
    """
    positions: dict[int, float] = {}
    counter = [0]

    def _assign(n: _MondrianNode, depth: int) -> float:
        if n.is_leaf() or (max_depth is not None and depth >= max_depth):
            x = float(counter[0])
            counter[0] += 1
            positions[id(n)] = x
            return x
        xl = _assign(n.left, depth + 1)
        xr = _assign(n.right, depth + 1)
        x = (xl + xr) / 2.0
        positions[id(n)] = x
        return x

    _assign(node, 0)
    return positions


def _draw_tree(
    tree: _MondrianNode,
    ax,
    node_label_fn,
    max_depth: int | None = None,
    dpi: int = 150,
) -> None:
    """Draw a node-link tree diagram on *ax*.

    Parameters
    ----------
    tree : _MondrianNode
        Root of the tree to draw.
    ax : matplotlib.axes.Axes
        Target axes.
    node_label_fn : callable
        ``node_label_fn(node, depth) -> str`` producing the label for each node.
    max_depth : int or None
        If given, collapse subtrees below this depth into leaf-like nodes.
    dpi : int
        Display DPI for rendering quality. Higher values produce sharper text
        and lines (default 150 for high-resolution output).

    Notes
    -----
    For publication-quality vector output, save the figure with ``bbox_inches='tight'``::

        >>> ax = clf.draw(backend="matplotlib")  # doctest: +SKIP
        >>> ax.figure.savefig("tree.svg", dpi=300, bbox_inches="tight")  # doctest: +SKIP
    """
    positions = _node_x_positions(tree, max_depth=max_depth)

    def _draw_node(n: _MondrianNode, depth: int) -> None:
        x = positions[id(n)]
        y = -depth

        is_collapsed = (not n.is_leaf()) and (max_depth is not None and depth >= max_depth)
        is_leaf = n.is_leaf() or is_collapsed

        # Box style
        boxstyle = "round,pad=0.3"
        facecolor = "#d4e6f1" if is_leaf else "#fdebd0"
        label = node_label_fn(n, depth, collapsed=is_collapsed)
        ax.text(
            x, y, label,
            ha="center", va="center", fontsize=7,
            bbox=dict(boxstyle=boxstyle, facecolor=facecolor, edgecolor="#555", linewidth=0.8),
            zorder=3,
        )

        if not is_leaf:
            for child in (n.left, n.right):
                cx = positions[id(child)]
                cy = -(depth + 1)
                ax.plot([x, cx], [y, cy], color="#555", linewidth=0.8, zorder=2)
            _draw_node(n.left, depth + 1)
            _draw_node(n.right, depth + 1)

    _draw_node(tree, 0)
    ax.axis("off")
    # Set high DPI for better quality rendering
    if hasattr(ax, 'figure'):
        ax.figure.dpi = dpi


def _partition_cells(node: _MondrianNode, lower: NDArray, upper: NDArray):
    """Yield (leaf_node, partition_lower, partition_upper) for every leaf.

    Unlike ``node.lower_bounds``/``upper_bounds`` (which store the *data-extent*
    bounding box of the points within a node), the partition bounds returned
    here represent the actual rectangular cell assigned by the Mondrian process.
    Each split propagates the cut boundary down, so the cells tile the full root
    bounding box with no gaps or overlaps.
    """
    if node.is_leaf():
        yield node, lower, upper
        return
    left_upper = upper.copy()
    left_upper[node.split_dim] = node.split_loc
    yield from _partition_cells(node.left, lower.copy(), left_upper)
    right_lower = lower.copy()
    right_lower[node.split_dim] = node.split_loc
    yield from _partition_cells(node.right, right_lower, upper.copy())


def _draw_partition(
    tree: _MondrianNode,
    X: NDArray,
    ax,
    leaf_color_fn,
    scatter_X: NDArray | None = None,
    scatter_y=None,
    scatter_cmap=None,
) -> None:
    """Draw the 2-D Mondrian box-partition for a tree with ``d == 2``.

    Parameters
    ----------
    tree : _MondrianNode
        Root of the tree.
    X : ndarray, shape (n, 2)
        Training features (used to determine clipping bounds for boundary cells).
    ax : matplotlib.axes.Axes
        Target axes.
    leaf_color_fn : callable
        ``leaf_color_fn(leaf) -> color`` mapping each leaf to a face colour.
    scatter_X : ndarray or None
        If given, scatter these points on top of the partition.
    scatter_y : array-like or None
        Labels / values for colouring the scatter points.
    scatter_cmap : str or Colormap or None
        Colormap for scatter points.
    """
    from matplotlib.patches import Rectangle

    margin = 0.05 * max(float(X[:, 0].max() - X[:, 0].min()), float(X[:, 1].max() - X[:, 1].min()), 1.0)
    x0_min, x0_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    x1_min, x1_max = X[:, 1].min() - margin, X[:, 1].max() + margin

    def _clip(v, lo, hi):
        return max(lo, min(hi, float(v)))

    # Use root's data-extent as the starting partition bounds so the cells
    # tile exactly that region.  Boundary cells extend to the clip limits.
    root_lower = tree.lower_bounds.copy()
    root_upper = tree.upper_bounds.copy()
    root_lower[0] = x0_min
    root_upper[0] = x0_max
    root_lower[1] = x1_min
    root_upper[1] = x1_max

    for leaf, lo_bounds, hi_bounds in _partition_cells(tree, root_lower, root_upper):
        lo0 = _clip(lo_bounds[0], x0_min, x0_max)
        hi0 = _clip(hi_bounds[0], x0_min, x0_max)
        lo1 = _clip(lo_bounds[1], x1_min, x1_max)
        hi1 = _clip(hi_bounds[1], x1_min, x1_max)

        if hi0 <= lo0 or hi1 <= lo1:
            continue

        color = leaf_color_fn(leaf)
        rect = Rectangle(
            (lo0, lo1), hi0 - lo0, hi1 - lo1,
            facecolor=color, edgecolor="#333", linewidth=0.6, alpha=0.6,
        )
        ax.add_patch(rect)

    if scatter_X is not None:
        ax.scatter(scatter_X[:, 0], scatter_X[:, 1], c=scatter_y, cmap=scatter_cmap,
                   s=18, linewidths=0.4, edgecolors="k", zorder=5)

    ax.set_xlim(x0_min, x0_max)
    ax.set_ylim(x1_min, x1_max)


# ---------------------------------------------------------------------------
# Unsupervised lifetime tuning helpers (Methods 1, 2, 3)
# ---------------------------------------------------------------------------

def _compute_subtree_counts(root: _MondrianNode) -> dict[int, int]:
    """Return a dict mapping id(node) → total sample count in that subtree.

    Uses a post-order DFS so each internal node's count equals the sum of its
    children's counts.  Leaf counts are ``len(node.indices)``.  The dict is
    keyed by ``id(node)`` (object identity) to avoid mutating the dataclass.
    """
    counts: dict[int, int] = {}

    def _dfs(node: _MondrianNode) -> None:
        if node.is_leaf():
            counts[id(node)] = len(node.indices) if node.indices is not None else 0
        else:
            _dfs(node.left)
            _dfs(node.right)
            counts[id(node)] = counts[id(node.left)] + counts[id(node.right)]

    _dfs(root)
    return counts


def _compute_partition_bounds(
    root: _MondrianNode,
    X_lower: NDArray,
    X_upper: NDArray,
) -> dict[int, tuple]:
    """Return a dict mapping id(node) → (cell_lower, cell_upper) for all nodes.

    Propagates the rectangular partition-cell bounds from root to leaves using a
    pre-order iterative DFS.  The root cell is initialised to
    (``X_lower``, ``X_upper``); each split clips the child bounds along the
    split dimension.

    Unlike ``node.lower_bounds``/``upper_bounds`` (which store the data-extent
    bounding box), the returned bounds represent the actual Mondrian partition
    cells and tile the root rectangle with no gaps or overlaps.
    """
    bounds: dict[int, tuple] = {}
    stack = [(root, X_lower.copy(), X_upper.copy())]
    while stack:
        node, lo, hi = stack.pop()
        bounds[id(node)] = (lo, hi)
        if not node.is_leaf():
            left_hi = hi.copy()
            left_hi[node.split_dim] = node.split_loc
            right_lo = lo.copy()
            right_lo[node.split_dim] = node.split_loc
            # Push right first so left is processed first (LIFO)
            stack.append((node.right, right_lo, hi.copy()))
            stack.append((node.left, lo.copy(), left_hi))
    return bounds


def _unsupervised_lifetime_sqrt_n(
    master_root: _MondrianNode,
    x_test: NDArray,
    n_total: int,
    subtree_counts: dict[int, int],
) -> float:
    """Method 1 — Targeted Leaf Capacity (√n heuristic).

    Descends the master tree toward ``x_test`` and halts at the first node
    whose subtree sample count is ≤ √n_total.  Returns the split_time of that
    node's *parent*, which becomes λ*.

    With this λ*, the actual tree (grown from the same RNG state after
    restoration by ``_resolve_lifetime``) will produce a leaf around
    ``x_test`` containing approximately √n_total samples — a natural
    bias-variance balance for kernel-density-style nonconformity scores.

    Args:
        master_root: Root of a fully grown master tree (lifetime = ∞).
        x_test: Test feature vector (d,).
        n_total: Size of the augmented bag |X_bag| = n + 1.
        subtree_counts: Precomputed dict from ``_compute_subtree_counts``.

    Returns:
        Optimal lifetime λ* ≥ 0.
    """
    target = n_total ** 0.5
    parent_split_time = 0.0
    current = master_root

    while not current.is_leaf():
        if subtree_counts[id(current)] <= target:
            # This node's subtree already holds ≤ √n samples; the parent's
            # split_time is the last "useful" cut — use it as λ*.
            return parent_split_time
        parent_split_time = current.split_time
        if x_test[current.split_dim] <= current.split_loc:
            current = current.left
        else:
            current = current.right

    # Reached a leaf before count ≤ target: every leaf has ≤ target samples
    # already; return the last internal node's split_time.
    return parent_split_time


def _unsupervised_lifetime_density(
    master_root: _MondrianNode,
    X_bag: NDArray,
    subtree_counts: dict[int, int],
    partition_bounds: dict[int, tuple],
) -> float:
    """Method 3 — Spatial Density Log-Likelihood.

    Sweeps through all realised split times of the master tree in ascending
    order, computing the incremental change in the Mondrian piecewise-uniform
    spatial log-likelihood at each event.  Returns the split time τ* that
    maximises the cumulative log-likelihood plus a small ε offset so that the
    winning split is *included* in the actual tree (since ``split_time ≥
    lifetime`` makes a leaf).

    The log-likelihood contribution of a cell l with n_l samples and volume
    V_l is n_l · log(n_l / V_l), dropping the constant −N log N term.
    Empty cells (n_l = 0) and zero-volume cells contribute 0.

    Complexity: O(k log k) in the number of internal nodes k ≤ n.

    Args:
        master_root: Root of a fully grown master tree (lifetime = ∞).
        X_bag: Augmented feature matrix (n + 1, d).
        subtree_counts: Precomputed dict from ``_compute_subtree_counts``.
        partition_bounds: Precomputed dict from ``_compute_partition_bounds``.

    Returns:
        Optimal lifetime λ* = τ* + 1e-10.  Returns 0.0 if no split improves
        the single-cell log-likelihood baseline.
    """
    # Dimensions with non-zero global range (guard against log(n / 0)).
    X_min = X_bag.min(axis=0)
    X_max = X_bag.max(axis=0)
    dims_ok = (X_max - X_min) > 0
    n_total = X_bag.shape[0]

    def _ll(count: int, lo: NDArray, hi: NDArray) -> float:
        """n_l · log(n_l / V_l), returning 0 for degenerate cases."""
        if count == 0:
            return 0.0
        widths = (hi - lo)[dims_ok]
        if len(widths) == 0:
            return 0.0
        vol = float(np.prod(widths))
        if vol <= 0.0:
            return 0.0
        return count * math.log(count / vol)

    # Collect internal nodes and sort by split_time (ascending).
    internal_nodes = [
        node for node, *_ in _iter_nodes(master_root) if not node.is_leaf()
    ]
    internal_nodes.sort(key=lambda nd: nd.split_time)

    if not internal_nodes:
        return 0.0

    # Baseline LL: whole X_bag as a single root cell.
    root_lo, root_hi = partition_bounds[id(master_root)]
    current_ll = _ll(n_total, root_lo, root_hi)
    best_ll = current_ll
    best_tau = 0.0

    for node in internal_nodes:
        n_node = subtree_counts[id(node)]
        lo_node, hi_node = partition_bounds[id(node)]

        n_left = subtree_counts[id(node.left)]
        lo_left, hi_left = partition_bounds[id(node.left)]

        n_right = subtree_counts[id(node.right)]
        lo_right, hi_right = partition_bounds[id(node.right)]

        delta = (
            _ll(n_left, lo_left, hi_left)
            + _ll(n_right, lo_right, hi_right)
            - _ll(n_node, lo_node, hi_node)
        )
        current_ll += delta

        if current_ll > best_ll:
            best_ll = current_ll
            best_tau = node.split_time

    # λ* = τ* + ε ensures the winning split is included (split_time < lifetime).
    return best_tau + 1e-10 if best_tau > 0.0 else 0.0


def _resolve_feature_weights(
    X_bag: NDArray,
    fw_spec: str | NDArray | None,
) -> NDArray | None:
    """Resolve the ``feature_weights`` parameter to a concrete weight array.

    Args:
        X_bag: Augmented feature matrix (n + 1, d).
        fw_spec: One of:

            ``None``
                No weighting — standard isotropic Mondrian.
            ``"variance"``
                Per-column population variance, normalised to sum 1.
                Zero-variance columns receive weight 0.
            ``NDArray`` of shape ``(d,)``
                User-supplied non-negative weight vector.

    Returns:
        Weight array of shape ``(d,)``, or ``None``.

    Raises:
        ValueError: For unrecognised strings or wrong-shape / negative arrays.
    """
    if fw_spec is None:
        return None

    if isinstance(fw_spec, str):
        if fw_spec == "variance":
            var = np.var(X_bag, axis=0, ddof=0)   # population variance, shape (d,)
            total = var.sum()
            if total == 0.0:
                # All features are constant — fall back to uniform weights.
                d = X_bag.shape[1]
                return np.full(d, 1.0 / d)
            return var / total
        raise ValueError(
            f"Unknown feature_weights string {fw_spec!r}. "
            "Recognised value: 'variance'."
        )

    fw = np.asarray(fw_spec, dtype=float)
    if fw.ndim != 1 or fw.shape[0] != X_bag.shape[1]:
        raise ValueError(
            f"feature_weights must have shape ({X_bag.shape[1]},); got {fw.shape}."
        )
    if np.any(fw < 0):
        raise ValueError("feature_weights must be non-negative.")
    return fw


def _resolve_lifetime(
    X_aug: NDArray,
    x_test: NDArray,
    lifetime_spec: float | str,
    rnd_gen: np.random.Generator,
    fw_arr: NDArray | None,
) -> float:
    """Resolve the ``lifetime`` parameter, running unsupervised tuning if needed.

    For string specifications a *master tree* is grown with ``lifetime = inf``
    using the current RNG state.  After λ* is computed, the RNG state is
    **restored** so that the actual tree drawn by the caller uses the exact same
    random sequence — making the actual tree a deterministic truncation of the
    master tree (identical splits up to λ*, different topology beyond it).

    This RNG restoration is mandatory: without it λ* would be computed from one
    spatial topology but applied to a completely different one, rendering the
    tuning meaningless.

    Args:
        X_aug: Augmented feature matrix ``[X_train ; x_test]``, shape (n+1, d).
        x_test: Test feature vector, shape (d,).
        lifetime_spec: A positive float, or one of ``"sqrt_n"`` / ``"density"``.
        rnd_gen: The model's random generator (state is snapshotted and restored).
        fw_arr: Resolved feature weight array (from ``_resolve_feature_weights``),
            or ``None`` for isotropic splits.

    Returns:
        Concrete lifetime value λ* ≥ 0.

    Raises:
        ValueError: For unrecognised string specifications.
    """
    if isinstance(lifetime_spec, (int, float)):
        return float(lifetime_spec)

    if lifetime_spec not in ("sqrt_n", "density"):
        raise ValueError(
            f"Unknown lifetime string {lifetime_spec!r}. "
            "Recognised values: 'sqrt_n', 'density'."
        )

    n_total = X_aug.shape[0]
    indices_all = np.arange(n_total)

    # 1. Snapshot the RNG state before growing the master tree.
    rng_state = rnd_gen.bit_generator.state

    # 2. Grow the master tree to completion (no lifetime cap).
    master = _sample_mondrian_tree(
        rnd_gen, X_aug, indices_all,
        parent_time=0.0, lifetime=np.inf,
        feature_weights=fw_arr,
    )

    # 3. Extract λ* from the master tree's topology.
    subtree_counts = _compute_subtree_counts(master)

    if lifetime_spec == "sqrt_n":
        lt_val = _unsupervised_lifetime_sqrt_n(master, x_test, n_total, subtree_counts)
    else:  # "density"
        X_lower = X_aug.min(axis=0)
        X_upper = X_aug.max(axis=0)
        partition_bounds = _compute_partition_bounds(master, X_lower, X_upper)
        lt_val = _unsupervised_lifetime_density(
            master, X_aug, subtree_counts, partition_bounds
        )

    # 4. Restore the RNG state — the actual tree must replay the same draws,
    #    producing a deterministic truncation of the master tree at λ*.
    rnd_gen.bit_generator.state = rng_state

    return lt_val


# ---------------------------------------------------------------------------
# Graphviz-backed tree visualisation helpers (optional dependency)
# ---------------------------------------------------------------------------

def _has_graphviz() -> bool:
    """Return True if the ``graphviz`` package *and* the ``dot`` binary are usable.

    The ``graphviz`` PyPI package is only a thin wrapper around the Graphviz
    system executables; rendering requires the ``dot`` binary on ``PATH``. We
    require both so that callers degrade gracefully to the matplotlib backend
    when the binary is missing instead of crashing inside ``dot``.
    """
    try:
        import graphviz  # noqa: F401
    except ImportError:
        return False
    import shutil
    return shutil.which("dot") is not None


def _dot_escape(s: str) -> str:
    """Escape a string for use as a DOT quoted label."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _build_graphviz_digraph(
    tree: _MondrianNode,
    node_label_fn,
    max_depth: int | None,
    *,
    title: str = "",
) -> graphviz.Digraph:
    """Build a :class:`graphviz.Digraph` for *tree*.

    Parameters
    ----------
    tree : _MondrianNode
        Root node.
    node_label_fn : callable
        ``node_label_fn(node, depth, *, collapsed=False) -> str``.
    max_depth : int or None
        If given, collapse subtrees below this depth into leaf-like boxes.
    title : str
        Graph title (rendered as graph label).

    Returns
    -------
    graphviz.Digraph
    """
    import graphviz

    dg = graphviz.Digraph(comment=title)
    dg.attr(
        rankdir="TB",
        label=_dot_escape(title),
        fontsize="10",
        labelloc="t",
        fontname="sans-serif",
    )
    dg.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fontsize="9",
        fontname="sans-serif",
        margin="0.12,0.06",
    )
    dg.attr("edge", arrowsize="0.6", fontsize="8", fontname="sans-serif")

    counter = [0]

    def _add_node(n: _MondrianNode, depth: int, parent_id: str | None = None, edge_label: str = "") -> None:
        node_id = f"n{counter[0]}"
        counter[0] += 1
        is_collapsed = (not n.is_leaf()) and (max_depth is not None and depth >= max_depth)
        is_leaf = n.is_leaf() or is_collapsed

        label = _dot_escape(node_label_fn(n, depth, collapsed=is_collapsed))
        fillcolor = "#d4e6f1" if is_leaf else "#fdebd0"
        dg.node(node_id, label=label, fillcolor=fillcolor)

        if parent_id is not None:
            dg.edge(parent_id, node_id, label=edge_label)

        if not is_leaf:
            _add_node(n.left,  depth + 1, node_id, edge_label=f"≤ {n.split_loc:.3g}")
            _add_node(n.right, depth + 1, node_id, edge_label=f"> {n.split_loc:.3g}")

    _add_node(tree, 0)
    return dg


def _render_tree(
    tree: _MondrianNode,
    ax,
    node_label_fn,
    max_depth: int | None,
    backend: str,
    title: str,
    dpi: int = 150,
):
    """Render a Mondrian tree diagram using the chosen *backend*.

    Parameters
    ----------
    tree : _MondrianNode
        Root node to render.
    ax : matplotlib.axes.Axes or None
        If given, draw into this axes.  If ``None`` and the graphviz backend is
        used, a :class:`graphviz.Digraph` is returned directly (renders as
        inline SVG in Jupyter notebooks).
    node_label_fn : callable
        ``node_label_fn(node, depth, *, collapsed=False) -> str``.
    max_depth : int or None
        Collapse depth; passed through to the renderer.
    backend : {'auto', 'graphviz', 'matplotlib'}
        ``'auto'`` uses graphviz if available, else falls back to matplotlib.
        ``'graphviz'`` raises :class:`ImportError` with an install hint if the
        package is missing.
        ``'matplotlib'`` forces the built-in renderer regardless of whether
        graphviz is installed.
    title : str
        Diagram title.

    Returns
    -------
    graphviz.Digraph or matplotlib.axes.Axes
        Returns a :class:`graphviz.Digraph` when graphviz is used and *ax* is
        ``None``; otherwise returns the target :class:`~matplotlib.axes.Axes`.

    Notes
    -----
    For publication-quality vector output with the graphviz backend, call
    ``draw(backend="graphviz")`` with no ``ax`` argument to get a
    :class:`graphviz.Digraph`, then use its ``.render()`` method (vector
    output via SVG/PDF)::

        >>> dg = clf.draw(backend="graphviz")  # doctest: +SKIP
        >>> dg.render("mytree", format="svg", cleanup=True)  # doctest: +SKIP

    When an ``ax`` is provided with the graphviz backend, the tree is embedded
    as a raster image which may appear blurry when saved. For vector output
    inside subplots, use ``backend="matplotlib"`` instead.
    """
    if backend == "graphviz":
        if not _has_graphviz():
            raise ImportError(
                "The 'graphviz' package is required for backend='graphviz'. "
                "Install it with:  pip install online-cp[viz]\n"
                "(The 'graphviz' pip package also needs the system 'dot' binary — "
                "install graphviz via your OS package manager, e.g. "
                "'apt install graphviz' or 'brew install graphviz'.)"
            )
        use_graphviz = True
    elif backend == "auto":
        use_graphviz = _has_graphviz()
    elif backend == "matplotlib":
        use_graphviz = False
    else:
        raise ValueError(
            f"Unknown backend {backend!r}. Choose from 'auto', 'graphviz', 'matplotlib'."
        )

    if use_graphviz:
        dg = _build_graphviz_digraph(tree, node_label_fn, max_depth, title=title)
        if ax is None:
            return dg
        # Embed as raster PNG into an existing Axes (e.g. for subplot composition).
        import io

        import matplotlib.image as mpimg

        # Set DPI via graph attributes for higher resolution output
        dg.attr(dpi=str(dpi))
        png_bytes = dg.pipe(format="png")
        img = mpimg.imread(io.BytesIO(png_bytes))
        ax = _get_ax(ax)
        ax.imshow(img)
        ax.axis("off")
        return ax

    # ---- matplotlib fallback ----
    ax = _get_ax(ax)
    _draw_tree(tree, ax, node_label_fn, max_depth=max_depth, dpi=dpi)
    ax.set_title(title, fontsize=9)
    return ax


# ---------------------------------------------------------------------------
# Forest summary builders (numpy-only, used by the ensemble adapters)
# ---------------------------------------------------------------------------


def _build_tree_summary(
    seed: int,
    X_aug: NDArray,
    y_train: NDArray,
    label_to_idx: dict,
    K: int,
    n: int,
    lifetime: float,
    max_depth: int | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Build one Mondrian tree and return lightweight numpy summaries.

    Builds the tree from X_aug (n+1 points), assigns training-only counts
    (test point's structural slot is excluded), then packs all per-point leaf
    information into pure numpy arrays so callers never need the tree objects.

    Parameters
    ----------
    seed : int
        Seed for this tree's RNG.
    X_aug : ndarray of shape (n+1, d)
        Augmented feature matrix (training + test point).
    y_train : ndarray of shape (n,)
        Training labels.
    label_to_idx : dict
        Mapping from label to integer index.
    K : int
        Number of classes.
    n : int
        Number of training points (test point is at index n).
    lifetime : float
        Mondrian tree depth budget.

    Returns
    -------
    n_leaves : ndarray of shape (n+1,)
        Size of each point's leaf (structural, includes test point's slot).
    counts_matrix : ndarray of shape (n+1, K)
        Training-only class counts of each point's leaf.
    leaf_star_train_indices : ndarray of int
        Indices (< n) of training points sharing the test point's leaf.
    """
    n_total = len(X_aug)
    rng = np.random.default_rng(seed)
    tree = _sample_mondrian_tree(rng, X_aug, np.arange(n_total), 0.0, lifetime, max_depth=max_depth)
    return _summarize_tree(tree, y_train, label_to_idx, K, n)


def _summarize_tree(
    root: _MondrianNode,
    y_train: NDArray,
    label_to_idx: dict,
    K: int,
    n: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """Pack per-point leaf classification info from a grown-or-extended tree.

    Assigns training-only counts (the test point's slot at index ``n`` is
    excluded) then returns the same summary as :func:`_build_tree_summary`.
    Shared by the batch forest (fresh trees) and the online forest (extensions
    of persistent trees).
    """
    _assign_counts(root, y_train, label_to_idx, K, n_train=n)
    n_total = n + 1

    n_leaves = np.empty(n_total, dtype=np.int64)
    counts_matrix = np.empty((n_total, K), dtype=np.int64)
    leaf_star_train_indices = np.array([], dtype=np.int64)

    for leaf in _collect_leaves(root):
        sz = leaf.n_points()
        is_leaf_star = False
        for idx in leaf.indices:
            n_leaves[idx] = sz
            counts_matrix[idx] = leaf.counts
            if int(idx) == n:
                is_leaf_star = True
        if is_leaf_star:
            leaf_star_train_indices = leaf.indices[leaf.indices < n]

    return n_leaves, counts_matrix, leaf_star_train_indices


def _build_tree_summary_reg(
    seed: int,
    X_aug: NDArray,
    y_train: NDArray,
    n: int,
    lifetime: float,
    max_depth: int | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Build one Mondrian tree and return regression leaf statistics.

    Parameters
    ----------
    seed : int
        RNG seed for this tree.
    X_aug : ndarray, shape (n+1, d)
        Augmented feature matrix (training + test point at index n).
    y_train : ndarray, shape (n,)
        Training labels.
    n : int
        Number of training points (test point is at index n).
    lifetime : float
        Mondrian depth budget.

    Returns
    -------
    base_ncm : ndarray, shape (n,)
        NCM for each training point: |y_i − μ_leaf_i| where μ is the
        training-only mean of the leaf.
    leaf_star_train_idx : ndarray of int
        Indices < n of training points sharing the test-point's leaf.
    leaf_star_mu_train : float
        Training-only mean of y in leaf_star (B/A, or 0.0 if A=0).
    """
    n_total = n + 1
    rng = np.random.default_rng(seed)
    tree = _sample_mondrian_tree(rng, X_aug, np.arange(n_total), 0.0, lifetime, max_depth=max_depth)
    return _summarize_tree_reg(tree, y_train, n, X_aug[n])


def _summarize_tree_reg(
    root: _MondrianNode,
    y_train: NDArray,
    n: int,
    x_test: NDArray,
) -> tuple[NDArray, NDArray, float]:
    """Regression leaf statistics from a grown-or-extended tree.

    Returns the same ``(base_ncm, leaf_star_train_idx, leaf_star_mu_train)``
    summary as :func:`_build_tree_summary_reg`. Shared by the batch forest
    (fresh trees) and the online forest (extensions of persistent trees).
    """
    leaf_star = _find_leaf(root, x_test)

    base_ncm = np.empty(n, dtype=float)
    leaf_star_train_idx = np.array([], dtype=np.int64)

    for leaf in _collect_leaves(root):
        train_idx = leaf.indices[leaf.indices < n]
        if len(train_idx) == 0:
            continue
        mu = y_train[train_idx].mean()
        base_ncm[train_idx] = np.abs(y_train[train_idx] - mu)
        if leaf is leaf_star:
            leaf_star_train_idx = train_idx

    A = len(leaf_star_train_idx)
    leaf_star_mu_train = float(y_train[leaf_star_train_idx].mean()) if A > 0 else 0.0

    return base_ncm, leaf_star_train_idx, leaf_star_mu_train


# ---------------------------------------------------------------------------
# MondrianTree: standalone object-oriented core
# ---------------------------------------------------------------------------


class MondrianTree:
    """A single sampled Mondrian partition of a data bag ``X``.

    ``MondrianTree`` owns one realisation of the Mondrian process at a given
    ``lifetime`` and exposes growing, traversal, leaf statistics and
    visualisation. Conformal and Venn predictors are thin adapters that grow a
    tree from an augmented bag and read off leaf statistics.

    The partition is a *bag function*: it depends only on the feature values in
    ``X``, never on their order. Growing with a string ``lifetime`` yields a
    deterministic truncation of the master tree grown with the same RNG.

    Parameters
    ----------
    root : _MondrianNode
        Root node of the sampled tree. Prefer :meth:`grow` over constructing
        directly.
    X : ndarray of shape (n, d)
        The data bag the tree was grown from.
    lifetime : float
        The concrete (resolved) lifetime the tree was grown at.
    feature_weights : ndarray of shape (d,) or None
        The resolved per-dimension split weights (``None`` for isotropic).

    Examples
    --------
    >>> import numpy as np
    >>> from online_cp.mondrian import MondrianTree
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(0, 1, size=(20, 2))
    >>> tree = MondrianTree.grow(X, rng, lifetime=2.0)
    >>> leaf = tree.find_leaf(X[0])
    >>> leaf.is_leaf()
    True
    >>> all(node.is_leaf() for node in tree.collect_leaves())
    True
    """

    def __init__(
        self,
        root: _MondrianNode,
        X: NDArray,
        *,
        lifetime: float,
        feature_weights: NDArray | None = None,
    ) -> None:
        self.root = root
        self.X = np.asarray(X)
        self.lifetime = lifetime
        self.feature_weights = feature_weights

    @classmethod
    def grow(
        cls,
        X: NDArray,
        rng: np.random.Generator,
        *,
        lifetime: float | str = 1.0,
        x_test: NDArray | None = None,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        feature_weights: str | NDArray | None = None,
        verbose: int = 0,
    ) -> MondrianTree:
        """Grow a Mondrian tree from the data bag ``X`` using ``rng``.

        The generator ``rng`` is advanced in place. String ``lifetime``
        specifications (``'sqrt_n'`` / ``'density'``) grow a master tree to
        resolve the budget and then restore the RNG state, so the returned tree
        is a deterministic truncation of that master tree.

        Parameters
        ----------
        X : ndarray of shape (n, d)
            Feature bag; the partition is a function of these values only.
        rng : numpy.random.Generator
            Random generator (advanced in place).
        lifetime : float or {'sqrt_n', 'density'}
            Mondrian time budget, or an unsupervised auto-tuning rule.
        x_test : ndarray of shape (d,), optional
            Reference point for the ``'sqrt_n'`` rule (defaults to the last row
            of ``X``, i.e. the augmented test point).
        max_depth : int or None
            Hard depth cap (``None`` = no cap).
        min_samples_leaf : int
            Minimum number of points allowed in a leaf.
        feature_weights : {'variance'} or array-like of shape (d,) or None
            Unsupervised per-dimension split weights.
        verbose : int
            Verbosity level.

        Returns
        -------
        MondrianTree
        """
        X = np.asarray(X)
        fw_arr = _resolve_feature_weights(X, feature_weights)
        ref = X[-1] if x_test is None else np.asarray(x_test).ravel()
        lt_val = _resolve_lifetime(X, ref, lifetime, rng, fw_arr)
        root = _sample_mondrian_tree(
            rng,
            X,
            np.arange(X.shape[0]),
            parent_time=0.0,
            lifetime=lt_val,
            verbose=verbose,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            feature_weights=fw_arr,
        )
        return cls(root, X, lifetime=lt_val, feature_weights=fw_arr)

    def extend(self, x: NDArray, rng: np.random.Generator) -> MondrianTree:
        """Return a new tree with ``x`` projected in (online ``ExtendMondrianBlock``).

        The result has the same law as :meth:`grow` on the augmented point set
        but costs only ``O(depth)`` (Roy & Teh 2009). This tree is left unchanged;
        ``x`` is appended as the last row of the returned tree's ``X``.

        Requires a fixed float ``lifetime`` (the projective regime); string
        lifetimes are not supported online.
        """
        if not isinstance(self.lifetime, (int, float)):
            raise ValueError(
                "extend() requires a fixed float lifetime; got "
                f"{self.lifetime!r}. String/auto-tuned lifetimes are batch-only."
            )
        x = np.asarray(x, dtype=float).ravel()
        x_idx = self.X.shape[0]
        new_root = _extend_mondrian(
            self.root, x, x_idx, rng, float(self.lifetime), self.feature_weights
        )
        return MondrianTree(
            new_root,
            np.vstack([self.X, x]),
            lifetime=self.lifetime,
            feature_weights=self.feature_weights,
        )

    def find_leaf(self, x: NDArray) -> _MondrianNode:
        """Return the leaf node whose cell contains ``x``."""
        return _find_leaf(self.root, np.asarray(x).ravel())

    def collect_leaves(self) -> list[_MondrianNode]:
        """Return all leaf nodes, left-to-right."""
        return _collect_leaves(self.root)

    def iter_nodes(self) -> list[tuple]:
        """Depth-first pre-order over ``(node, depth, parent_id, node_id)`` tuples."""
        return _iter_nodes(self.root)

    def struct_stats(self) -> dict:
        """Return structural stats: ``n_nodes``, ``n_leaves``, ``n_branches``, ``height``."""
        return _tree_struct_stats(self.root)

    def assign_counts(
        self,
        y: NDArray,
        label_to_idx: dict,
        K: int,
        n_train: int | None = None,
    ) -> None:
        """Attach per-class count vectors to every leaf (modifies the tree in place)."""
        _assign_counts(self.root, y, label_to_idx, K, n_train=n_train)

    def to_dataframe(self):
        """Export the tree structure as a :class:`pandas.DataFrame` (one row per node)."""
        import pandas as pd

        rows = []
        node_id_map: dict[str, int] = {}
        next_id = 0
        for node, depth, parent_id_str, node_id_str in _iter_nodes(self.root):
            if node_id_str not in node_id_map:
                node_id_map[node_id_str] = next_id
                next_id += 1
            node_id_int = node_id_map[node_id_str]
            if parent_id_str is None:
                parent_id_int = None
            else:
                if parent_id_str not in node_id_map:
                    node_id_map[parent_id_str] = next_id
                    next_id += 1
                parent_id_int = node_id_map[parent_id_str]
            rows.append({
                "node_id": node_id_int,
                "parent_id": parent_id_int,
                "is_leaf": node.is_leaf(),
                "depth": depth,
                "split_dim": node.split_dim,
                "split_loc": float(node.split_loc) if not node.is_leaf() else None,
                "split_time": float(node.split_time) if not node.is_leaf() else None,
                "parent_time": float(node.parent_time),
                "n_points": node.n_points(),
            })
        return pd.DataFrame(rows)

    def draw(self, ax=None, max_depth=None, backend="auto"):
        """Draw a node-link diagram of the tree.

        See :meth:`ConformalMondrianTreeClassifier.draw
        <online_cp.classifiers.ConformalMondrianTreeClassifier.draw>` for the
        backend semantics. Uses a structural label (split test / leaf size).

        Notes
        -----
        For publication-quality vector output:

        * **Graphviz** (when ``ax=None``): Returns a :class:`graphviz.Digraph`
          which can be saved as SVG/PDF without quality loss::

              >>> dg = tree.draw(backend="graphviz")  # doctest: +SKIP
              >>> dg.render("mytree", format="svg", cleanup=True)  # doctest: +SKIP

        * **Matplotlib**: Use ``backend="matplotlib"`` and save with
          ``bbox_inches='tight'``::

              >>> ax = tree.draw(backend="matplotlib")  # doctest: +SKIP
              >>> ax.figure.savefig("tree.svg", dpi=300, bbox_inches="tight")  # doctest: +SKIP
        """
        def label_fn(node, depth, *, collapsed=False):
            if node.is_leaf() or collapsed:
                n = node.n_points()
                return f"leaf\nn={n}" if n else "empty"
            return f"x[{node.split_dim}] ≤\n{node.split_loc:.3g}"

        return _render_tree(
            self.root, ax, label_fn, max_depth, backend, title="Mondrian Tree",
        )

    def draw_partition(self, ax=None, scatter=True):
        """Draw the 2-D box partition (requires ``d == 2``), one colour per leaf."""
        if self.X.shape[1] != 2:
            raise ValueError(
                f"draw_partition() requires exactly 2 features, got {self.X.shape[1]}."
            )
        from matplotlib import colormaps

        ax = _get_ax(ax)
        leaves = self.collect_leaves()
        cmap = colormaps["tab20"].resampled(max(len(leaves), 1))
        leaf_order = {id(leaf): i for i, leaf in enumerate(leaves)}

        def leaf_color_fn(leaf):
            return cmap(leaf_order[id(leaf)] % cmap.N)

        _draw_partition(
            self.root, self.X, ax, leaf_color_fn,
            scatter_X=self.X if scatter else None,
        )
        return ax

    def __len__(self) -> int:
        return _tree_struct_stats(self.root)["n_leaves"]

    def __repr__(self) -> str:
        stats = _tree_struct_stats(self.root)
        return (
            f"MondrianTree(lifetime={self.lifetime!r}, "
            f"n_leaves={stats['n_leaves']}, height={stats['height']})"
        )


__all__ = ["MondrianTree"]
