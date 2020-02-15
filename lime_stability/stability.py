"""
Main script, where Lime classes are redefined in order to calculate stability indices.
To do so, the original class LimeTabularExplainer gets overridden by the LimeTabularExplainerOvr,
which allows for the same usage done in original Lime.
Moreover it contains the check_stability method, which calculates CSI and VSI indices.
"""

from lime.lime_tabular import LimeTabularExplainer
from lime.lime_base import LimeBase
import numpy as np
from sklearn.linear_model import Ridge

from lime_stability.utils import refactor_confints_todict, compute_WLS_stdevs, compare_confints, LocalModelError


class LimeBaseOvr(LimeBase):
    """Override of the original LimeBase class in lime_base"""

    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None):
        """Inherits from the original LimeBase class"""

        super(LimeBaseOvr, self).__init__(kernel_fn, verbose, random_state)

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='auto',
                                   model_regressor=None):
        """
        Override of the original explain_instance_with_data method,
        done to have hooks for the generated dataset and other quantities,
        important in order to calculate confidence intervals and to check stability.



        Original Documentation of the function:

        Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            (intercept, exp, score, local_pred):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
            local_pred is the prediction of the explanation model on the original instance
        """

        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection)
        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=self.random_state)
        easy_model = model_regressor
        easy_model.fit(neighborhood_data[:, used_features],
                       labels_column, sample_weight=weights)
        prediction_score = easy_model.score(
            neighborhood_data[:, used_features],
            labels_column, sample_weight=weights)

        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))

        # Hooks added to extract important info, as class attributes
        try:
            assert isinstance(easy_model, Ridge)
        except AssertionError:
            self.alpha = None
            print("""Attention: Lime Local Model is not a Weighted Ridge Regression (WRR),
            Lime Method will work anyway, but the stability indices may not be computed
            (the formula is model specific)""")
        else:
            self.alpha = easy_model.alpha
        finally:
            self.easy_model = easy_model
            self.X = neighborhood_data[:, used_features]
            self.weights = weights
            self.true_labels = labels_column

        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred, )
            print('Right:', neighborhood_labels[0, label])
        return (easy_model.intercept_,
                sorted(zip(used_features, easy_model.coef_),
                       key=lambda x: np.abs(x[1]), reverse=True),
                prediction_score, local_pred)


class LimeTabularExplainerOvr(LimeTabularExplainer):
    """Override of the original LimeTabularExplainer class in lime_tabular

        A new method has been implemented: check_stability
        This method calculates the VSI and CSI indices for a Lime trained instance"""

    def __init__(self,
                 training_data,
                 mode="classification",
                 training_labels=None,
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 kernel_width=None,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 discretize_continuous=True,
                 discretizer='quartile',
                 sample_around_instance=False,
                 random_state=None,
                 training_data_stats=None):
        """Inherits from the original LimeTabularExplainer class"""

        super(LimeTabularExplainerOvr, self).__init__(training_data, mode, training_labels, feature_names,
                                                      categorical_features, categorical_names, kernel_width,
                                                      kernel, verbose, class_names, feature_selection,
                                                      discretize_continuous, discretizer, sample_around_instance,
                                                      random_state, training_data_stats)

        # Override base attribute, with an instance of LimeBaseOvr class
        kernel_fn = self.base.kernel_fn
        self.base = LimeBaseOvr(kernel_fn, verbose=verbose, random_state=self.random_state)

    def confidence_intervals(self):
        """
    Method to calculate stability indices of an application of the LIME method
    to a particular unit of the dataset.
    The stability indices are described in the paper:
    "Statistical stability indices for LIME: obtaining reliable explanations for Machine Learning models"
    which can be found in the ArXiv online repository: https://arxiv.org/abs/2001.11757
    The paper is currently under review in the Journal of Operational Research Society (JORS)

    Returns:
        conf_int: list of two values (lower and upper bound of the confidence interval)
    """

        stdevs_beta = compute_WLS_stdevs(X=self.base.X, Y=self.base.true_labels,
                                         weights=self.base.weights, alpha=self.base.alpha)

        beta_ridge = self.base.easy_model.coef_.tolist()
        conf_int = refactor_confints_todict(means=beta_ridge, st_devs=stdevs_beta, feat_names=self.feature_names)

        return conf_int

    def check_stability(self,
                        data_row,
                        predict_fn,
                        labels=(1,),
                        top_labels=None,
                        num_features=10,
                        num_samples=5000,
                        distance_metric='euclidean',
                        model_regressor=None,
                        n_calls=10,
                        index_verbose=False,
                        verbose=False):

        """
        Method to calculate stability indices for a trained LIME instance.
        The stability indices are relative to the particular data point we are explaining.
        The stability indices are described in the paper:
        "Statistical stability indices for LIME: obtaining reliable explanations for Machine Learning models".
        It can be found in the ArXiv online repository: https://arxiv.org/abs/2001.11757
        The paper is currently under review in the Journal of Operational Research Society (JORS)

        Args:
            data_row: 1d numpy array or scipy.sparse matrix, corresponding to a row
            predict_fn: prediction function. For classifiers, this should be a
                function that takes a numpy array and outputs prediction
                probabilities. For regressors, this takes a numpy array and
                returns the predictions. For ScikitClassifiers, this is
                `classifier.predict_proba()`. For ScikitRegressors, this
                is `regressor.predict()`. The prediction function needs to work
                on multiple feature vectors (the vectors randomly perturbed
                from the data_row).
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase.
                If not Ridge Regression, the stability indices may not be calculated
                and the method will raise a LocalModelError.
            n_calls: number of repeated Lime calls to perform.
                High number of calls slows down the execution,
                however a meaningful comparison takes place only when
                there is a reasonable number of calls to compare.
            index_verbose: Controls for the verbosity at the stability indices level,
                when set to True gives information about partial values related to stability.
            verbose: Controls for the verbosity at the LocalModel level,
                when set to True, gives information about the repeated calls of WRR.

        Returns:
            csi: index to evaluate the stability of the coefficients of each variable across
            different Lime explanations obtained from the repeated n_calls.
            Ranges from 0 to 100.
            vsi: index to evaluate whether the variables retrieved in different Lime explanations
                are the same. Ranges from 0 to 100.
        """

        # Override verbosity in the LimeBaseOvr instance
        previous_verbose = self.base.verbose
        self.base.verbose = verbose

        confidence_intervals = []
        for i in range(n_calls):
            self.explain_instance(data_row,
                                  predict_fn,
                                  labels,
                                  top_labels,
                                  num_features,
                                  num_samples,
                                  distance_metric,
                                  model_regressor)

            # The first time check if the local model is WRR, otherwise raise Exception
            if not i:
                if self.base.alpha is None:
                    raise LocalModelError("""Lime Local Model is not a Weighted Ridge Regression (WRR),
                    Stability indices may not be computed: the formula is model specific""")

            confidence_intervals.append(self.confidence_intervals())

        csi, vsi = compare_confints(confidence_intervals=confidence_intervals,
                                    index_verbose=index_verbose)

        # Set back the original verbosity
        self.base.verbose = previous_verbose

        return csi, vsi
