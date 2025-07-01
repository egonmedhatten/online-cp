# This file should contain the code to keep track of evaluation metrics
import numpy as np
import matplotlib.pyplot as plt

# NOTE: Below are some efficiency criteria, some of which are proper scoring rules.
#       What I have to do now, is to create one or more Evaluation classes, which, in their
#       construction create metric classes, and keeps track of the statistics as we go on.

import numpy as np

# This is a bit messy...
class Evaluation:
    """
    Class to evaluate and keep track of various metrics.
    """

    def __init__(self, metrics=None, **kwargs):
        """
        Initialize the Evaluation class with specified metrics.

        Parameters:
        metrics (list): List of metric names to be tracked.
        kwargs (dict): Additional metric classes to be tracked.
        """
        self.metrics = {}
        self.results = {}

        self.n = 0

        available_metrics = {
            "oe": OE,
            "of": OF,
            "err": Err,
            "width": Width,
            "winkler": WinklerScore,
            "crps": CRPS,
        }

        if metrics is not None:
            for metric_name in metrics:
                if metric_name.lower() in available_metrics:
                    self._add_metric_instance(metric_name.lower(), available_metrics[metric_name.lower()])
                else:
                    raise ValueError(f"Unknown metric: {metric_name}")

        for metric_name, metric_class in kwargs.items():
            if isinstance(metric_class, type) and issubclass(metric_class, (OE, OF, Err, Width, WinklerScore, CRPS)):
                self._add_metric_instance(metric_name.lower(), metric_class)
            elif metric_name.lower() in available_metrics:
                self._add_metric_instance(metric_name.lower(), available_metrics[metric_name.lower()])
            else:
                raise ValueError(f"Unknown metric: {metric_name}")

    def _add_metric_instance(self, name, metric_class):
        """
        Add an instance of a metric class to the evaluation.

        Parameters:
        name (str): Name of the metric.
        metric_class (class): Class of the metric.
        """
        self.metrics[name] = metric_class()
        self.results[name] = []

    def update(self, y, Gamma=None, p_values=None, cpd=None, raise_errors=True):
        """
        Update the metrics with new data.

        Parameters:
        y: True label.
        Gamma: Prediction set.
        p_values: Probability values.
        cpd: Conformal predictive distribution.
        epsilon: Significance level.
        raise_errors (bool): Whether to raise errors for missing data.
        """
        for name, metric in self.metrics.items():
            try:
                if isinstance(metric, OE) or isinstance(metric, Err) or isinstance(metric, Width) or isinstance(metric, WinklerScore):
                    if Gamma is None:
                        if raise_errors:
                            raise ValueError(f"Gamma (gamma_val) is required for {name}.")
                        else:
                            continue

                    if isinstance(metric, WinklerScore):
                        result = metric._update(Gamma, y, Gamma.epsilon)  # Correct arguments for WinklerScore
                    elif isinstance(metric, Width):
                        result = metric._update(Gamma) # Correct arguments for Width
                    else:
                        result = metric._update(y, Gamma)      # Correct arguments for OE, Err
                elif isinstance(metric, OF):
                    if p_values is None:
                        if raise_errors:
                            raise ValueError(f"Probability values (p_vals) are required for {name}.")
                        else:
                            continue
                    result = metric._update(p_values, y)           # Correct arguments for OF
                elif isinstance(metric, CRPS):
                    if cpd is None:
                        if raise_errors:
                            raise ValueError(f"CPD (cpd) is required for {name}.")
                        else:
                            continue
                    result = metric._update(cpd, y)              # Correct arguments for CRPS
                else:
                    raise ValueError(f"Unknown metric type: {type(metric)}")

                self.results[name].append(result)  # Store the result

            except ValueError as e:
                if raise_errors:
                    raise
                else:
                    print(f"Warning: Skipping metric '{name}': {e}")
        self.n += 1
        
    def summarize(self):
        """
        Summarize the results of the metrics.

        Returns:
        dict: Summary of the metrics.
        """
        summary = {}
        for name, results in self.results.items():
            summary[name] = {
                "mean": np.mean(results) if results else None,
                "std": np.std(results) if results else None,
                "min": np.min(results) if results else None,
                "max": np.max(results) if results else None,
                "last": results[-1] if results else None,
            }
        return summary
    
    def cumulative(self, criterion):
        """
        Calculate the cumulative sum of a criterion.

        Parameters:
        criterion (str): Name of the criterion.

        Returns:
        np.ndarray: Cumulative sum of the criterion.
        """
        return np.cumsum(self.results[criterion])
    
    def __getattr__(self, name):  # Called when an attribute is not found
        """
        Get a metric by name.

        Parameters:
        name (str): Name of the metric.

        Returns:
        Metric instance.
        """
        if name in self.metrics:
            return self.metrics[name]
        else:
            raise AttributeError(f"Metric '{name}' not found in results.")

    def __dir__(self): #For autocompletion
        """
        List all available metrics.

        Returns:
        list: List of metric names.
        """
        return list(self.metrics.keys())
    
    
    # FIXME: I want to have x tics on the x-axis, but the same x-axis for all plots
    def plot_cumulative_results(self, max_cols=2, title=None):
        """
        Plot the cumulative results of the metrics.

        Parameters:
        max_cols (int): Maximum number of columns in the plot.
        title (str): Title of the plot.

        Returns:
        matplotlib.figure.Figure: Figure object of the plot.
        """
        num_criteria = len(self.results)
        
        # Determine grid size
        num_cols = min(max_cols, num_criteria)  # Limit columns
        num_rows = int(np.ceil(num_criteria / num_cols))

        # Adaptive figure size
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 3 * num_rows), squeeze=False, sharex=False)

        # Flatten axs for easy indexing if it's multidimensional
        axs = axs.flatten()

        for i, (criterion, values) in enumerate(self.results.items()):
            cs = np.where(np.isinf(values), np.nan, np.cumsum(values))
            axs[i].plot(cs)
            # axs[i].set_title(criterion)
            axs[i].set_xlabel('Number of examples')
            axs[i].set_ylabel(criterion)

        # Hide any unused subplots
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        if title is not None:
            fig.suptitle(title)
        fig.tight_layout()
        plt.close(fig)  # Prevent implicit display
        return fig


class EfficiencyCriterion:
    """
    Base class for efficiency criteria.
    """

    def __init__(self):
        """
        Initialize the EfficiencyCriterion class.
        """
        self.value = 0.0
        self.n = 0

    def mean(self):
        """
        Calculate the mean value of the criterion.

        Returns:
        float: Mean value.
        """
        return self.value / self.n

class OE(EfficiencyCriterion):
    """
    Observed excess (OE) is the number of incorrect labels included in the prediction set Gamma.
    Thus, it depends on the significance level varepsilon.
    OE is a conditionally proper efficiency criterion (see Algorithmic Learning in a Random World 2nd edition).
    """
    def __init__(self):
        """
        Initialize the OE class.
        """
        super().__init__()

    def _update(self, y, Gamma):
        """
        Update the OE metric with new data.

        Parameters:
        y: True label.
        Gamma: Prediction set.

        Returns:
        int: Updated OE value.
        """
        self.n += 1
        if y in Gamma:
            oe = len(Gamma) - 1
        else:
            oe = len(Gamma)
        self.value += oe
        return oe
    
class OF(EfficiencyCriterion):
    """
    Observed fuzzyness (OF) is the sum of incorrect p-values. Thus, it is independent of the significance level varepsilon.
    OF is a conditionally proper efficiency criterion (see Algorithmic Learning in a Random World 2nd edition).
    """
    def __init__(self):
        """
        Initialize the OF class.
        """
        super().__init__()

    def _update(self, p_values, y):
        """
        Update the OF metric with new data.

        Parameters:
        p_values (dict): p-values.
        y: True label.

        Returns:
        float: OF value.
        """
        self.n += 1
        of = 0
        for label, p in p_values.items():
            if not label == y:
                of += p
        self.value += of
        return of
    
class Err(EfficiencyCriterion):
    """
    Err evaluates whether the true label is included in the prediction set Gamma.
    """

    def __init__(self):
        """
        Initialize the Err class.
        """
        super().__init__()

    def _update(self, y, Gamma):
        """
        Update the Err metric with new data.

        Parameters:
        y: True label.
        Gamma: Prediction set.

        Returns:
        int: Updated Err value.
        """
        self.n += 1
        if not type(Gamma) == tuple:
            err = int(not(y in Gamma))
        else:       
            err = int(not(y in Gamma))
        self.value += err
        return err

    
class Width(EfficiencyCriterion):
    """
    The width of the prediciton interval Gamma.
    """
    def __init__(self):
        """
        Initialize the Width class.
        """
        super().__init__()
    
    def _update(self, Gamma):
        """
        Update the Width metric with new data.

        Parameters:
        Gamma: Prediction set.

        Returns:
        float: Width value.
        """
        self.n += 1
        w = Gamma.width()
        self.value += w
        return w
    
class WinklerScore(EfficiencyCriterion):
    """
    The Winkler interval score (or just interval score) is a proper scoring rule for evaluation interval forecasts.
    """

    def __init__(self):
        """
        Initialize the WinklerScore class.
        """
        super().__init__()

    def _update(self, Gamma, y, epsilon=0.1):
        """
        Update the WinklerScore metric with new data.

        Parameters:
        Gamma: Prediction set.
        y: True label.
        epsilon (float): Significance level.

        Returns:
        float: WinklerScore value.
        """
        epsilon = Gamma.epsilon
        self.n += 1
        try:
            assert Gamma.upper < np.inf
            assert Gamma.lower > -np.inf
        except AssertionError:
            ws = np.inf
            self.value += ws
            return ws
        l = Gamma.lower
        u = Gamma.upper
        base_score = u - l
        penalty = 0
        
        if y < l:
            penalty = (2 / epsilon) * (l - y)
        elif y > u:
            penalty = (2 / epsilon) * (y - u)
        
        ws = base_score + penalty

        self.value += ws
        return ws
    
class CRPS(EfficiencyCriterion):
    """
    Continuous ranked probability score (CRPS) is a proper scoring rule for evaluating predictive distributions.
    In the case of conformal predictive distributions, the definition does not work, as the integral will not 
    converge. To address this, we have pieced together a confergent integral by integrating the lower distribution
    when x <= y, and the upper when x > y.
    """

    def __init__(self):
        """
        Initialize the CRPS class.
        """
        super().__init__()

    def _update(self, cpd, y):
        """
        Update the CRPS metric with new data.

        Parameters:
        cpd: Conformal predictive distribution.
        y: True label.

        Returns:
        float: CRPS value.
        """
        '''
        NOTE: This implementation is a bith shaky, really. I integrate the lower distribution for x <= y, 
              and the upper for x > y, to ensure convergence. It is possible that this is not proper.
        '''
        self.n += 1
        func_ad_hoc = lambda x: (cpd(x, 0) - int(y <= x))**2 if x <= y else (cpd(x, 1) - int(y <= x))**2

        crps = np.trapz([func_ad_hoc(x) for x in cpd.y_vals[1:-1]], cpd.y_vals[1:-1]) # NOTE: Perhaps not the best integrator

        self.value += crps
        return crps
    
if __name__ == "__main__":
    import doctest
    import sys
    (failures, _) = doctest.testmod()
    if failures:
        sys.exit(1)
