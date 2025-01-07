# To do
1. p-values for CRR is not correct. We need
    * Upper p-value
    * Lower p-value
    * Combined p-value
2. Enable predictions at multiple significance levels
3. LSPM
4. Mondrian (how to do it?)

# Methods to implement?

## Regression
* Conformalized ridge regression (CRR)
* Deleted conformalized ridge regression (deleted CRR)
* Studentized  conformalizedridge regression (studentized CRR)
* (Bayesian ridge regression as soft model)
* Conformalized k-nearest neighbours regression (CkNNR)
* Conformalized lasso regression (CLasso)
* Conformalized elastic net regression (CEN)

Also kernelized version of all of the above. It is a bit unfortunate that the $l2$ versions (least squares, ridge) have to be handled differently with updates etc (linear algebra formula "goes in the other direction"). Try to be economical with the code.

## Probabilistic regression
* Least squared prediction machine (LSPM)
* Studentized least squared prediction machine (studentized LSPM)
* Kernel ridge regression prediction machine (KRRPM)
* Nearest neighbours prediction machine (kNNPM)

## Decision making
* Reject option in (binary) classification
* Predictive decision making system (PDMS)

## Mondrian CP
Nice to have. Think about it.

## Classification
* k nearest neighbours classifier (kNNC)
* Support vector machine (SVM)
* Figure out some others. Scoring type classifiers are nice to work with. Figure out if Passive-aggressive works.

## Test martingales
* Plugin martingale (several kinds are possible, e.g. kernel density, empirical distribution, perhaps even parametric betting functions)
* Simple jumper
* Composite jumper
* Sleepy jumper
* Others?

## Online compression models
Might be nice to implement online compression models to play around with. Low priority.
* Exchangeability model
* Gaussian model
* Gauss linear model
* Gauss linear exchangeability model
* Markov model
* Hypergraphical model

## Repetitive structures
Similar motivation to online compression models. Could be nice to play around with.


-----------------------
# Future considerations

### Release minimal version 
For use in projects, it may be good to have a released minimal version of OnlineConformalPrediction. Initially, it could include
* Conformalised Ridge Regression
* Plugin martingale
* Possibly Conformalised Nearest Neighbours Regression (but I will have to check it carefully for any bugs)

### Properties of CPs?
* Should we keep track of errors internally in the parent class? 
* Should we store the average interval size?
* For classifiers; should we store the efficiency metrics?

### Linear regression
We will initally focus on regression, but online classification is actually easier. A simple class that uses e.g. scikit-learn classifiers to define nonconformity measure could be easily implemented. 

There are at least three commonly used regularisations used in linear regression, all of which are compatible with the kernel trick. 
* $L1$ (Lasso)
* $L2$ (Ridge)
* Linear combination of the above (Elastic net)

All of these can be conformalized, and at least Ridge can also be used in conformal predictive systems (CPS).

Another relatively simple regressor is the k-nearest neighbours algorithm, which is very flexible as it can use arbitrary distances. It is particularly interesting in the CPS setting. The distance can be measured in feature space as defined by a kernel.

Ridge and KNN are described in detail in Algorithmic Learning in a Random World. Lasso and Elastic net are conformalised in the paper Fast Exact Conformalization of Lasso using Piecewise Linear Homotopy, but I am unaware of any extention to CPS. 

### Teaching schedule
Section 3.3 in Algorithmic Learning in a Radnom World deals with, so called, weak teachers. In the pure online mode, labels arrive immediately after a predition is made. This makes little sense in practice. The notion of a teaching schedule formalises this, and makes the relevant validity guarantees clear. There are three types of validity; weak, strong, and iterated logartihm validity. 

There may be settings where the user wants to specify a teaching schedule beforehand, to guarantee some property of validity. It may also be the case that the teaching schedule is implied by the usage, and it would then be useful to know if the resulting prediciton sets are valid.

A teaching schedule also serves as documentation of what has been done, which could be useful in practice.

## Todo
* Should we add some scaler? Don't know if it is neccesary for Ridge
* Possibly add a class MimoConformalRidgeRegressor
* Add CPS version of ridge regressor?
* Possibly add a TeachingSchedule?
* Possibly add ACI, both for single, and MIMO CRR?
* Add references to papers and books to README
* Add k-NN regressor and CPS