# Regression
* Conformalized ridge regression (CRR)
* Deleted conformalized ridge regression (deleted CRR)
* Studentized  conformalizedridge regression (studentized CRR)
* (Bayesian ridge regression as soft model)
* Conformalized k-nearest neighbours regression (CkNNR)
* Conformalized lasso regression (CLasso)
* Conformalized elastic net regression (CEN)

Also kernelized version of all of the above. It is a bit unfortunate that the $l2$ versions (least squares, ridge) have to be handled differently with updates etc (linear algebra formula "goes in the other direction"). Try to be economical with the code.

# Probabilistic regression
* Least squared prediction machine (LSPM)
* Studentized least squared prediction machine (studentized LSPM)
* Kernel ridge regression prediction machine (KRRPM)
* Nearest neighbours prediction machine (kNNPM)

# Decision making
* Reject option in (binary) classification
* Predictive decision making system (PDMS)

# Mondrian CP
Nice to have. Think about it.

# Classification
* k nearest neighbours classifier (kNNC)
* Support vector machine (SVM)
* Figure out some others. Scoring type classifiers are nice to work with. Figure out if Passive-aggressive works.

# Test martingales
* Plugin martingale (several kinds are possible, e.g. kernel density, empirical distribution, perhaps even parametric betting functions)
* Simple jumper
* Composite jumper
* Sleepy jumper
* Others?

# Online compression models
Might be nice to implement online compression models to play around with. Low priority.
* Exchangeability model
* Gaussian model
* Gauss linear model
* Gauss linear exchangeability model
* Markov model
* Hypergraphical model

# Repetitive structures
Similar motivation to online compression models. Could be nice to play around with.