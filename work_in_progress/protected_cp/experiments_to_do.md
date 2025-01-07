# Experiments

Here, we collect the experiments we need to perform for the paper. Ideally, each experiment should be performed both for full CP online and ICP offline. However, it may be excessive. Perhaps if we figured out how to protect ICP regression in general, and stick to that, as it is the most popular one. 

## Classification
The simplest case to handle. We have three obvious datasets, all rather large
1. USPS
2. Wine
3. Satlog

Since computations are expensive, it suffices to save the p-values for each class and process them later.

Efficiency criterion either OE or OF.

## Regression
No real good dataset to use. Possibly the ferry data that Simon used. It has a change point. 

A (possibly suitable alternative, is one of the housing datasets)

Of course, it would be awsome to get it to work on time series, but that may be a stretch. Unless we can come up with a decent betting scheme based on periodicity or something... In which case, electricity demand data could be nice.

What is the efficiency criterion to use. Winkler's interval score?

## Conformal predictive systems
This is a regression problem, so again, we have no good dataset on offer.

Again, efficiency criteria is difficult. CRPS is difficult, as CPD, or rather RPD in this case, is not a proper CDF.

# Presentation
We must figure out how to present the results. 