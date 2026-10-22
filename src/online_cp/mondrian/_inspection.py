"""Shared inspection & visualisation for the Mondrian tree/forest predictors.

The four conformal Mondrian adapters (``ConformalMondrianTree*`` /
``ConformalMondrianForest*``) share identical ``summary`` / ``to_dataframe`` /
``debug_one`` / ``draw`` / ``draw_partition`` behaviour; they differ only in the
per-leaf *content* (classifiers describe leaves by class counts, regressors by
the leaf mean/variance) and, for ``draw_partition``, in the legend vs. colourbar
decoration.

``_MondrianInspectionMixin`` holds the shared logic and delegates the
task-specific parts to small hooks. ``_MondrianClassifierInspection`` and
``_MondrianRegressorInspection`` supply those hooks; the adapters simply inherit
the appropriate one. The mixin is duck-typed against the adapter attributes
(``_last_tree``, ``X``, ``y``, ``lifetime``, ``epsilon``, ``rnd_state``,
``max_depth`` and, for classifiers, ``label_space``).
"""

from __future__ import annotations

import numpy as np

from online_cp.mondrian.tree import (
    MondrianTree,
    _draw_partition,
    _get_ax,
    _iter_nodes,
    _render_tree,
    _tree_struct_stats,
)


class _MondrianInspectionMixin:
    """Shared ``summary`` / ``to_dataframe`` / ``debug_one`` / ``draw`` /
    ``draw_partition`` for the Mondrian conformal adapters."""

    # Overridden by task mixins.
    _draw_title: str = "Mondrian Tree"

    # ------------------------------------------------------------------
    # Task hooks (supplied by subclasses)
    # ------------------------------------------------------------------
    def _summary_task_fields(self) -> dict:
        """Task-specific ``summary`` fields (class info vs. y statistics)."""
        raise NotImplementedError

    def _leaf_row(self, node) -> dict:
        """Extra ``to_dataframe`` columns for a leaf node."""
        return {}

    def _leaf_desc(self, node) -> str:
        """Trailing per-leaf description for ``debug_one`` (may be empty)."""
        return ""

    def _leaf_draw_text(self, node, n: int) -> str:
        """Leaf label for ``draw`` (``n`` is the leaf's point count, ``> 0``)."""
        return f"n={n}"

    def _render_partition(self, ax, scatter: bool) -> None:
        """Colour the leaves and decorate the axes for ``draw_partition``."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared implementation
    # ------------------------------------------------------------------
    def _require_tree(self) -> None:
        if self._last_tree is None:
            raise RuntimeError(
                "No tree has been built yet. Call predict() or compute_p_value() first."
            )

    @property
    def summary(self) -> dict:
        """Summary statistics of the model (and of the last cached tree, if any)."""
        stats = {
            "n_points": len(self.y) if self.y is not None else 0,
            "n_features": self.X.shape[1] if self.X is not None else 0,
        }
        stats.update(self._summary_task_fields())
        stats["lifetime"] = self.lifetime
        stats["epsilon"] = self.epsilon
        if hasattr(self, "n_trees"):
            stats["n_trees"] = self.n_trees
        if self._last_tree is not None:
            stats.update(_tree_struct_stats(self._last_tree))
            stats["total_observed_weight"] = len(self.y)
        return stats

    def to_dataframe(self):
        """Export the last cached tree as a :class:`pandas.DataFrame` (one row per node).

        Columns: ``node_id``, ``parent_id``, ``is_leaf``, ``depth``,
        ``split_dim``, ``split_loc``, ``split_time``, ``parent_time``,
        ``bbox_lower``, ``bbox_upper``, ``n_points``, plus task-specific
        leaf columns (``counts`` for classifiers; ``y_mean`` / ``y_std`` for
        regressors).

        Raises
        ------
        RuntimeError
            If no tree has been built yet (call ``predict()`` first).
        """
        self._require_tree()
        import pandas as pd

        rows = []
        node_id_map: dict[str, int] = {}
        next_id = 0
        for node, depth, parent_id_str, node_id_str in _iter_nodes(self._last_tree):
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

            bbox_lower = node.lower_bounds.tolist() if len(node.lower_bounds) > 0 else []
            bbox_upper = node.upper_bounds.tolist() if len(node.upper_bounds) > 0 else []
            row = {
                "node_id": node_id_int,
                "parent_id": parent_id_int,
                "is_leaf": node.is_leaf(),
                "depth": depth,
                "split_dim": node.split_dim,
                "split_loc": float(node.split_loc) if not node.is_leaf() else None,
                "split_time": float(node.split_time) if not node.is_leaf() else None,
                "parent_time": float(node.parent_time),
                "bbox_lower": bbox_lower,
                "bbox_upper": bbox_upper,
                "n_points": node.n_points(),
            }
            if node.is_leaf():
                row.update(self._leaf_row(node))
            rows.append(row)
        return pd.DataFrame(rows)

    def debug_one(self, x):
        """Trace one example through a representative tree, returning a path string.

        Builds a fresh representative tree from a deterministic seed (this does
        not touch the model's RNG) and traces ``x`` through it, showing each
        split decision and the final leaf statistics.
        """
        x = np.asarray(x, dtype=float).ravel()
        if self.X is None:
            raise RuntimeError("Must call learn_initial_training_set first")

        rng_debug = np.random.default_rng(self.rnd_state if self.rnd_state is not None else 0)
        X_aug = np.vstack([self.X, x])
        tree = MondrianTree.grow(
            X_aug,
            rng_debug,
            lifetime=self.lifetime,
            x_test=X_aug[-1],
            max_depth=self.max_depth,
            feature_weights=getattr(self, "feature_weights", None),
        )

        path_lines = []
        current = tree.root
        depth = 0
        while not current.is_leaf():
            split_str = f"  {'  ' * depth}x[{current.split_dim}] <= {current.split_loc:.6f}"
            if x[current.split_dim] <= current.split_loc:
                path_lines.append(split_str + "  [TRUE → left]")
                current = current.left
            else:
                path_lines.append(split_str + "  [FALSE → right]")
                current = current.right
            depth += 1

        leaf_str = f"  {'  ' * depth}LEAF: n_points={current.n_points()}" + self._leaf_desc(current)
        path_lines.append(leaf_str)
        return "\n".join(path_lines)

    def draw(self, ax=None, max_depth=None, backend="auto", **kwargs):
        """Draw a node-link diagram of the last cached tree.

        Internal nodes show the split test ``x[i] ≤ threshold``; leaves show
        task-specific statistics (majority class / purity for classifiers, leaf
        mean / spread for regressors); empty cells are labelled ``"empty"``.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Target axes. If ``None`` and the graphviz backend is active, a
            :class:`graphviz.Digraph` is returned (renders inline in notebooks).
        max_depth : int or None
            Display-only depth cap (does not affect the model).
        backend : {'auto', 'graphviz', 'matplotlib'}
            Rendering backend; ``'auto'`` uses graphviz when both the package and
            the ``dot`` binary are available, else matplotlib.

        Raises
        ------
        RuntimeError
            If no tree has been built yet.

        Notes
        -----
        For publication-quality vector output:

        * **Graphviz** (when ``ax=None``): Returns a :class:`graphviz.Digraph`
          which can be saved as SVG/PDF without quality loss::

              >>> dg = clf.draw(backend="graphviz")  # doctest: +SKIP
              >>> dg.render("mytree", format="svg", cleanup=True)  # doctest: +SKIP

        * **Matplotlib**: Use ``backend="matplotlib"`` and save with
          ``bbox_inches='tight'``::

              >>> ax = clf.draw(backend="matplotlib")  # doctest: +SKIP
              >>> ax.figure.savefig("tree.svg", dpi=300, bbox_inches="tight")  # doctest: +SKIP

        When an ``ax`` is provided with the graphviz backend, the tree is
        embedded as a raster image which may appear blurry when saved. For
        vector output inside subplots, use ``backend="matplotlib"`` instead.
        """
        self._require_tree()

        def label_fn(node, depth, *, collapsed=False):
            if node.is_leaf() or collapsed:
                n = node.n_points()
                if n == 0:
                    return "empty"
                return self._leaf_draw_text(node, n)
            return f"x[{node.split_dim}] ≤\n{node.split_loc:.3g}"

        return _render_tree(
            self._last_tree, ax, label_fn, max_depth, backend, title=self._draw_title,
        )

    def draw_partition(self, ax=None, scatter=True, **kwargs):
        """Draw the 2-D Mondrian box-partition of the last cached tree.

        Each leaf is drawn as a rectangle coloured by its leaf statistic;
        training points can be overlaid.

        Raises
        ------
        RuntimeError
            If no tree has been built yet.
        ValueError
            If the number of features is not exactly 2.
        """
        self._require_tree()
        if self.X.shape[1] != 2:
            raise ValueError(
                f"draw_partition() requires exactly 2 features, got {self.X.shape[1]}. "
                "The leaf bounding boxes only tile feature space exactly at d=2."
            )
        ax = _get_ax(ax)
        self._render_partition(ax, scatter)
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[1]")
        return ax


class _MondrianClassifierInspection(_MondrianInspectionMixin):
    """Leaf-content hooks for the Mondrian *classifier* adapters."""

    _draw_title = "Mondrian Tree (classifier)"

    def _summary_task_fields(self) -> dict:
        return {
            "n_classes": len(self.label_space) if self.label_space is not None else 0,
            "label_space": list(self.label_space) if self.label_space is not None else [],
        }

    def _leaf_row(self, node) -> dict:
        if node.counts is not None:
            return {
                "counts": {
                    self.label_space[i]: int(node.counts[i])
                    for i in range(len(self.label_space))
                }
            }
        return {}

    def _leaf_desc(self, node) -> str:
        if node.counts is not None:
            counts = {
                self.label_space[i]: int(node.counts[i])
                for i in range(len(self.label_space))
            }
            return f", counts={counts}"
        return ""

    def _leaf_draw_text(self, node, n: int) -> str:
        if node.counts is not None:
            maj_idx = int(np.argmax(node.counts))
            maj_label = self.label_space[maj_idx]
            counts_str = "  ".join(
                f"{self.label_space[i]}:{int(node.counts[i])}"
                for i in range(len(self.label_space))
                if node.counts[i] > 0
            )
            purity = int(node.counts[maj_idx]) / n
            return f"\u2192 class {maj_label}  ({purity:.0%})\nn={n}  [{counts_str}]"
        return f"n={n}"

    def _render_partition(self, ax, scatter: bool) -> None:
        from matplotlib import colormaps
        from matplotlib.lines import Line2D

        K = len(self.label_space)
        cmap = colormaps["tab10"].resampled(K)
        label_to_color = {lbl: cmap(i) for i, lbl in enumerate(self.label_space)}

        def leaf_color_fn(leaf):
            if leaf.counts is None or leaf.counts.sum() == 0:
                return "#cccccc"
            majority_idx = int(np.argmax(leaf.counts))
            return label_to_color[self.label_space[majority_idx]]

        scatter_X = self.X if scatter else None
        scatter_y = [list(self.label_space).index(yi) for yi in self.y] if scatter else None
        _draw_partition(
            self._last_tree, self.X, ax, leaf_color_fn,
            scatter_X=scatter_X, scatter_y=scatter_y, scatter_cmap="tab10",
        )
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=label_to_color[lbl],
                   markersize=8, label=str(lbl))
            for lbl in self.label_space
        ]
        ax.legend(handles=handles, title="Class", fontsize=7, title_fontsize=7)
        ax.set_title("Mondrian Partition (classifier)", fontsize=9)


class _MondrianRegressorInspection(_MondrianInspectionMixin):
    """Leaf-content hooks for the Mondrian *regressor* adapters."""

    _draw_title = "Mondrian Tree (regressor)"

    def _summary_task_fields(self) -> dict:
        has_y = self.y is not None and len(self.y) > 0
        return {
            "y_mean": float(np.mean(self.y)) if has_y else None,
            "y_std": float(np.std(self.y)) if has_y else None,
        }

    def _leaf_row(self, node) -> dict:
        if node.indices is not None:
            train_idx = node.indices[node.indices < len(self.y)]
            if len(train_idx) > 0:
                y_leaf = self.y[train_idx]
                return {"y_mean": float(np.mean(y_leaf)), "y_std": float(np.std(y_leaf))}
            return {"y_mean": None, "y_std": None}
        return {}

    def _leaf_desc(self, node) -> str:
        if node.indices is not None:
            train_idx = node.indices[node.indices < len(self.y)]
            if len(train_idx) > 0:
                y_leaf = self.y[train_idx]
                return f", y_mean={np.mean(y_leaf):.6f}, y_std={np.std(y_leaf):.6f}"
        return ""

    def _leaf_draw_text(self, node, n: int) -> str:
        if node.indices is not None:
            train_idx = node.indices[node.indices < len(self.y)]
            if len(train_idx) > 0:
                mu = float(np.mean(self.y[train_idx]))
                sd = float(np.std(self.y[train_idx]))
                cv_str = f"  CV={sd / abs(mu):.2f}" if abs(mu) > 1e-9 else ""
                return f"μ = {mu:.3g}\nn={n}  σ={sd:.3g}{cv_str}"
        return f"n={n}"

    def _render_partition(self, ax, scatter: bool) -> None:
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        from matplotlib import colormaps

        y_min, y_max = float(self.y.min()), float(self.y.max())
        norm = mcolors.Normalize(vmin=y_min, vmax=y_max)
        cmap = colormaps["coolwarm"]

        def leaf_color_fn(leaf):
            if leaf.indices is None:
                return "#cccccc"
            train_idx = leaf.indices[leaf.indices < len(self.y)]
            if len(train_idx) == 0:
                return "#cccccc"
            mu = float(np.mean(self.y[train_idx]))
            return cmap(norm(mu))

        scatter_X = self.X if scatter else None
        scatter_y = self.y if scatter else None
        _draw_partition(
            self._last_tree, self.X, ax, leaf_color_fn,
            scatter_X=scatter_X, scatter_y=scatter_y, scatter_cmap="coolwarm",
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="y mean")
        ax.set_title("Mondrian Partition (regressor)", fontsize=9)
