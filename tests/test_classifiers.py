import numpy as np

from online_cp.classifiers import ConformalNearestNeighboursClassifier


class TestConformalNearestNeighboursClassifier:
    def test_p_values_in_unit_interval(self, classification_dataset):
        X, y = classification_dataset
        label_space = np.unique(y)
        cp = ConformalNearestNeighboursClassifier(k=3, label_space=label_space, rnd_state=0)
        cp.learn_initial_training_set(X[:10], y[:10])

        for obj, lab in zip(X[10:30], y[10:30]):
            _, p_values = cp.predict(obj, return_p_values=True)
            cp.learn_one(obj, lab)
            for p in p_values.values():
                assert 0 <= p <= 1

    def test_validity(self, classification_dataset):
        """Error rate should be approximately <= epsilon on iid data."""
        X, y = classification_dataset
        label_space = np.unique(y)
        epsilon = 0.2
        cp = ConformalNearestNeighboursClassifier(k=3, label_space=label_space, rnd_state=1, epsilon=epsilon)

        n_init = 10
        cp.learn_initial_training_set(X[:n_init], y[:n_init])

        errors = 0
        n_test = len(y) - n_init
        for obj, lab in zip(X[n_init:], y[n_init:]):
            Gamma = cp.predict(obj)
            errors += int(lab not in Gamma)
            cp.learn_one(obj, lab)

        error_rate = errors / n_test
        # Allow generous margin for small sample
        assert error_rate <= epsilon + 0.10, f"Error rate {error_rate:.3f} exceeds epsilon={epsilon} + margin"

    def test_learn_one_grows_state(self):
        cp = ConformalNearestNeighboursClassifier(k=1, rnd_state=0)
        cp.learn_one(np.array([1.0, 2.0]), 1)
        assert cp.X.shape[0] == 1
        assert cp.y.shape[0] == 1

        cp.learn_one(np.array([3.0, 4.0]), -1)
        assert cp.X.shape[0] == 2
        assert cp.y.shape[0] == 2

    def test_prediction_set_nonempty_after_training(self, classification_dataset):
        """After sufficient training, prediction sets should not be empty at low epsilon."""
        X, y = classification_dataset
        label_space = np.unique(y)
        cp = ConformalNearestNeighboursClassifier(k=3, label_space=label_space, rnd_state=2, epsilon=0.01)
        cp.learn_initial_training_set(X[:50], y[:50])

        for obj in X[50:60]:
            Gamma = cp.predict(obj)
            assert len(Gamma) >= 1

    def test_first_prediction_includes_all_labels(self):
        """With no training data, all labels should be predicted."""
        cp = ConformalNearestNeighboursClassifier(k=1, label_space=np.array([0, 1, 2]), rnd_state=0)
        Gamma = cp.predict(np.array([0.0, 0.0]))
        assert len(Gamma) == 3
