import numpy as np
from scipy.stats import norm


class SimpleBayesClassifier:
    """
    A simple Naive Bayes Classifier that can use either histogram-based or Gaussian-based likelihood estimation.
    """

    def __init__(self, n_pos, n_neg):
        """
        Initialize the classifier with the number of positive and negative samples.

        Args:
            n_pos: Number of positive class samples (attrition = 1)
            n_neg: Number of negative class samples (attrition = 0)
        """
        self.n_pos = n_pos
        self.n_neg = n_neg

        # Calculate prior probabilities
        total = n_pos + n_neg
        if total == 0:
            self.prior_pos = 0.5
            self.prior_neg = 0.5
        else:
            self.prior_pos = n_pos / total
            self.prior_neg = n_neg / total

        # Parameters for histogram-based approach
        self.stay_params = None  # For negative class (attrition = 0)
        self.leave_params = None  # For positive class (attrition = 1)

        # Parameters for Gaussian approach
        self.stay_gaussian_params = None
        self.leave_gaussian_params = None

        self.n_bins = 10  # Default number of bins for discretization

    def fit_params(self, x, y):
        """
        Fit histogram-based parameters for each feature.

        Args:
            x: Training features (n_samples, n_features)
            y: Training labels (n_samples,)

        Returns:
            stay_params: List of (bins, edges) tuples for negative class
            leave_params: List of (bins, edges) tuples for positive class
        """
        n_features = x.shape[1]
        stay_params = []
        leave_params = []

        # Separate data by class
        x_stay = x[y == 0]  # Negative class (no attrition)
        x_leave = x[y == 1]  # Positive class (attrition)

        for feature_idx in range(n_features):
            # Get feature values for both classes (excluding NaN)
            feature_stay = x_stay[:, feature_idx]
            feature_leave = x_leave[:, feature_idx]

            # Remove NaN values for edge calculation
            feature_stay_clean = feature_stay[~np.isnan(feature_stay)]
            feature_leave_clean = feature_leave[~np.isnan(feature_leave)]

            # Combine to get overall range
            all_values = np.concatenate([feature_stay_clean, feature_leave_clean])

            if len(all_values) == 0:
                # No valid values, create dummy bins
                edges = np.array([0, 1])
                bins_stay = np.array([1])
                bins_leave = np.array([1])
            else:
                # Create bin edges
                min_val = np.min(all_values)
                max_val = np.max(all_values)

                # Add small epsilon to max to ensure all values are included
                edges = np.linspace(min_val, max_val + 1e-10, self.n_bins + 1)

                # Create histograms with Laplace smoothing (add-one smoothing)
                bins_stay, _ = np.histogram(feature_stay_clean, bins=edges)
                bins_leave, _ = np.histogram(feature_leave_clean, bins=edges)

                # Apply Laplace smoothing to avoid zero probabilities
                bins_stay = bins_stay + 1
                bins_leave = bins_leave + 1

            stay_params.append((bins_stay, edges))
            leave_params.append((bins_leave, edges))

        self.stay_params = stay_params
        self.leave_params = leave_params

        return stay_params, leave_params

    def fit_gaussian_params(self, x, y):
        """
        Fit Gaussian parameters (mean and std) for each feature.

        Args:
            x: Training features (n_samples, n_features)
            y: Training labels (n_samples,)

        Returns:
            stay_params: List of (mean, std) tuples for negative class
            leave_params: List of (mean, std) tuples for positive class
        """
        n_features = x.shape[1]
        stay_params = []
        leave_params = []

        # Separate data by class
        x_stay = x[y == 0]
        x_leave = x[y == 1]

        for feature_idx in range(n_features):
            # Get feature values for both classes (excluding NaN)
            feature_stay = x_stay[:, feature_idx]
            feature_leave = x_leave[:, feature_idx]

            # Remove NaN values
            feature_stay_clean = feature_stay[~np.isnan(feature_stay)]
            feature_leave_clean = feature_leave[~np.isnan(feature_leave)]

            # Calculate mean and std for stay class
            if len(feature_stay_clean) > 0:
                mean_stay = np.mean(feature_stay_clean)
                std_stay = np.std(feature_stay_clean)
                if std_stay == 0:
                    std_stay = 1e-10  # Avoid division by zero
            else:
                mean_stay = 0
                std_stay = 1

            # Calculate mean and std for leave class
            if len(feature_leave_clean) > 0:
                mean_leave = np.mean(feature_leave_clean)
                std_leave = np.std(feature_leave_clean)
                if std_leave == 0:
                    std_leave = 1e-10
            else:
                mean_leave = 0
                std_leave = 1

            stay_params.append((mean_stay, std_stay))
            leave_params.append((mean_leave, std_leave))

        self.stay_gaussian_params = stay_params
        self.leave_gaussian_params = leave_params

        return stay_params, leave_params

    def predict(self, x, threshold=0):
        """
        Predict using histogram-based Naive Bayes.

        Args:
            x: Test features (n_samples, n_features)
            threshold: Decision threshold (default=0 for log-likelihood ratio)

        Returns:
            predictions: Binary predictions (n_samples,)
        """
        log_likelihood_ratios = self._compute_log_likelihood_ratio(
            x, use_gaussian=False
        )
        predictions = (log_likelihood_ratios > threshold).astype(int)
        return predictions

    def gaussian_predict(self, x, threshold=0):
        """
        Predict using Gaussian-based Naive Bayes.

        Args:
            x: Test features (n_samples, n_features)
            threshold: Decision threshold (default=0 for log-likelihood ratio)

        Returns:
            predictions: Binary predictions (n_samples,)
        """
        log_likelihood_ratios = self._compute_log_likelihood_ratio(x, use_gaussian=True)
        predictions = (log_likelihood_ratios > threshold).astype(int)
        return predictions

    def _compute_log_likelihood_ratio(self, x, use_gaussian=False):
        """
        Compute log likelihood ratio for classification.

        Args:
            x: Test features (n_samples, n_features)
            use_gaussian: Whether to use Gaussian or histogram-based likelihood

        Returns:
            log_likelihood_ratios: Array of log likelihood ratios
        """
        n_samples = x.shape[0]
        n_features = x.shape[1]

        # Initialize log probabilities
        log_prob_leave = np.full(n_samples, np.log(self.prior_pos))
        log_prob_stay = np.full(n_samples, np.log(self.prior_neg))

        if use_gaussian:
            params_stay = self.stay_gaussian_params
            params_leave = self.leave_gaussian_params
        else:
            params_stay = self.stay_params
            params_leave = self.leave_params

        # Calculate log likelihood for each feature
        for feature_idx in range(n_features):
            feature_values = x[:, feature_idx]

            if use_gaussian:
                # Gaussian likelihood
                mean_stay, std_stay = params_stay[feature_idx]
                mean_leave, std_leave = params_leave[feature_idx]

                # Handle non-NaN values
                valid_mask = ~np.isnan(feature_values)

                if np.any(valid_mask):
                    # Calculate log probabilities using normal distribution
                    log_prob_stay[valid_mask] += norm.logpdf(
                        feature_values[valid_mask], mean_stay, std_stay
                    )
                    log_prob_leave[valid_mask] += norm.logpdf(
                        feature_values[valid_mask], mean_leave, std_leave
                    )
            else:
                # Histogram-based likelihood
                bins_stay, edges_stay = params_stay[feature_idx]
                bins_leave, edges_leave = params_leave[feature_idx]

                # Normalize bins to get probabilities
                prob_stay = bins_stay / np.sum(bins_stay)
                prob_leave = bins_leave / np.sum(bins_leave)

                # Digitize feature values to bin indices
                valid_mask = ~np.isnan(feature_values)

                if np.any(valid_mask):
                    bin_indices = (
                        np.digitize(feature_values[valid_mask], edges_stay) - 1
                    )
                    # Clip to valid range
                    bin_indices = np.clip(bin_indices, 0, len(prob_stay) - 1)

                    log_prob_stay[valid_mask] += np.log(prob_stay[bin_indices] + 1e-10)
                    log_prob_leave[valid_mask] += np.log(
                        prob_leave[bin_indices] + 1e-10
                    )

        # Return log likelihood ratio: log(P(leave|x)) - log(P(stay|x))
        return log_prob_leave - log_prob_stay

    def predict_proba_scores(self, x, use_gaussian=False):
        """
        Get prediction scores (log likelihood ratios) for ROC curve analysis.

        Args:
            x: Test features
            use_gaussian: Whether to use Gaussian or histogram-based likelihood

        Returns:
            scores: Log likelihood ratios
        """
        return self._compute_log_likelihood_ratio(x, use_gaussian=use_gaussian)
