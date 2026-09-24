# Mondrian Prediction

The Mondrian block groups everything built on the **Mondrian process** — a
stochastic process that recursively partitions feature space into axis-aligned
boxes. Because the partition depends only on the *bag* of feature values (never
their order), a tree grown from the augmented bag `{train ∪ (x_test, y)}` yields
an exact conformal predictor, and its leaves form a valid Venn taxonomy
(ALRW2 §2.2.9, §4.6, Thm 6.4).

!!! note "Two distinct meanings of *Mondrian*"
    - **Mondrian tree / forest predictors** (this page) use the Mondrian
      *process* to discover an adaptive partition and score nonconformity in the
      leaves.
    - The **group-conditional wrappers** (`MondrianConformalClassifier` /
      `MondrianConformalRegressor`) are the general *Mondrian taxonomy*
      framework — a user-supplied category function applied on top of any base
      predictor. They are documented at the bottom of this page.

## MondrianTree (core)

The standalone partition object. Grow it from a data bag with an external RNG,
then traverse it, read leaf statistics, or visualise it. Conformal and Venn
predictors are thin adapters over this class.

::: online_cp.mondrian.MondrianTree

## Conformal tree & forest predictors

::: online_cp.classifiers.ConformalMondrianTreeClassifier

::: online_cp.classifiers.ConformalMondrianForestClassifier

::: online_cp.regressors.ConformalMondrianTreeRegressor

::: online_cp.regressors.ConformalMondrianForestRegressor

## Mondrian Venn predictor

Uses the Mondrian leaf as the Venn taxonomy category (a bag function → Venn
validity, ALRW2 Thm 6.4).

::: online_cp.venn.MondrianVennPredictor

## Group-conditional wrappers

The general Mondrian *taxonomy* framework: a single pooled model with the
calibration step restricted to same-category examples (Vovk et al., 2005).

::: online_cp.mondrian.framework.MondrianConformalRegressor

::: online_cp.mondrian.framework.MondrianConformalClassifier
