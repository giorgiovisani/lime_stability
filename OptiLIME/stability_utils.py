"""
Main script, where Lime classes are redefined in order to calculate stability indices.
To do so, the original class LimeTabularExplainer gets overridden by the LimeTabularExplainerOvr,
which allows for the same usage done in original Lime.
Moreover it contains the check_stability method, which calculates CSI and VSI indices.
"""

from lime.lime_tabular import LimeTabularExplainer,TableDomainMapper
from lime.lime_base import LimeBase
from sklearn.linear_model import Ridge
from numpy.linalg import LinAlgError
import copy
import warnings
import numpy as np
import scipy as sp
import sklearn
import sklearn.preprocessing
from lime import explanation

# from NHANES1.lime_stability_original.utils import refactor_confints_todict, compute_WLS_stdevs, compare_confints, LocalModelError
from lime_stability.utils import refactor_confints_todict, compute_WLS_stdevs, compare_confints, LocalModelError


class LimeBaseOvr(LimeBase):
    """Override of the original LimeBase class in lime_base"""

    def __init__(self,
                 kernel_fn,
                 penalty,
                 verbose=False,
                 random_state=None):
        """Inherits from the original LimeBase class"""

        super(LimeBaseOvr, self).__init__(kernel_fn, verbose, random_state)
        self.penalty = penalty

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
            model_regressor = Ridge(alpha=self.penalty, fit_intercept=True,
                                    random_state=self.random_state)
        easy_model = model_regressor
        easy_model.fit(neighborhood_data[:, used_features],
                       labels_column, sample_weight=weights)
        prediction_score = easy_model.score(
            neighborhood_data[:, used_features],
            labels_column, sample_weight=weights)

        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))
        local_preds = easy_model.predict(neighborhood_data[:, used_features])
        self.ridge_pred = local_preds

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
                 penalty=1,
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
        self.penalty = penalty
        self.base = LimeBaseOvr(kernel_fn, verbose=verbose, random_state=self.random_state,penalty=self.penalty)

    #Override explain_instance to return also Lime Ridge predictions on the generated points
    def explain_instance(self,
                         data_row,
                         predict_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='euclidean',
                         model_regressor=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

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
                to Ridge regression in LimeBase. Must have model_regressor.coef_
                and 'sample_weight' as a parameter to model_regressor.fit()

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        if sp.sparse.issparse(data_row) and not sp.sparse.isspmatrix_csr(data_row):
            # Preventative code: if sparse, convert to csr format if not in csr format already
            data_row = data_row.tocsr()
        data, inverse = self.__data_inverse(data_row, num_samples)
        if sp.sparse.issparse(data):
            # Note in sparse case we don't subtract mean since data would become dense
            scaled_data = data.multiply(self.scaler.scale_)
            # Multiplying with csr matrix can return a coo sparse matrix
            if not sp.sparse.isspmatrix_csr(scaled_data):
                scaled_data = scaled_data.tocsr()
        else:
            scaled_data = (data - self.scaler.mean_) / self.scaler.scale_
        distances = sklearn.metrics.pairwise_distances(
                scaled_data,
                scaled_data[0].reshape(1, -1),
                metric=distance_metric
        ).ravel()

        yss = predict_fn(inverse)

        # for classification, the model needs to provide a list of tuples - classes
        # along with prediction probabilities
        if self.mode == "classification":
            if len(yss.shape) == 1:
                raise NotImplementedError("LIME does not currently support "
                                          "classifier models without probability "
                                          "scores. If this conflicts with your "
                                          "use case, please let us know: "
                                          "https://github.com/datascienceinc/lime/issues/16")
            elif len(yss.shape) == 2:
                if self.class_names is None:
                    self.class_names = [str(x) for x in range(yss[0].shape[0])]
                else:
                    self.class_names = list(self.class_names)
                if not np.allclose(yss.sum(axis=1), 1.0):
                    warnings.warn("""
                    Prediction probabilties do not sum to 1, and
                    thus does not constitute a probability space.
                    Check that you classifier outputs probabilities
                    (Not log probabilities, or actual class predictions).
                    """)
            else:
                raise ValueError("Your model outputs "
                                 "arrays with {} dimensions".format(len(yss.shape)))

        # for regression, the output should be a one-dimensional array of predictions
        else:
            try:
                if len(yss.shape) != 1 and len(yss[0].shape) == 1:
                    yss = np.array([v[0] for v in yss])
                assert isinstance(yss, np.ndarray) and len(yss.shape) == 1
            except AssertionError:
                raise ValueError("Your model needs to output single-dimensional \
                    numpyarrays, not arrays of {} dimensions".format(yss.shape))

            predicted_value = yss[0]
            min_y = min(yss)
            max_y = max(yss)

            # add a dimension to be compatible with downstream machinery
            yss = yss[:, np.newaxis]

        feature_names = copy.deepcopy(self.feature_names)
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row.shape[0])]

        if sp.sparse.issparse(data_row):
            values = self.convert_and_round(data_row.data)
            feature_indexes = data_row.indices
        else:
            values = self.convert_and_round(data_row)
            feature_indexes = None

        for i in self.categorical_features:
            if self.discretizer is not None and i in self.discretizer.lambdas:
                continue
            name = int(data_row[i])
            if i in self.categorical_names:
                name = self.categorical_names[i][name]
            feature_names[i] = '%s=%s' % (feature_names[i], name)
            values[i] = 'True'
        categorical_features = self.categorical_features

        discretized_feature_names = None
        if self.discretizer is not None:
            categorical_features = range(data.shape[1])
            discretized_instance = self.discretizer.discretize(data_row)
            discretized_feature_names = copy.deepcopy(feature_names)
            for f in self.discretizer.names:
                discretized_feature_names[f] = self.discretizer.names[f][int(
                        discretized_instance[f])]

        domain_mapper = TableDomainMapper(feature_names,
                                          values,
                                          scaled_data[0],
                                          categorical_features=categorical_features,
                                          discretized_feature_names=discretized_feature_names,
                                          feature_indexes=feature_indexes)
        ret_exp = explanation.Explanation(domain_mapper,
                                          mode=self.mode,
                                          class_names=self.class_names)
        if self.mode == "classification":
            ret_exp.predict_proba = yss[0]
            if top_labels:
                labels = np.argsort(yss[0])[-top_labels:]
                ret_exp.top_labels = list(labels)
                ret_exp.top_labels.reverse()
        else:
            ret_exp.predicted_value = predicted_value
            ret_exp.min_value = min_y
            ret_exp.max_value = max_y
            labels = [0]
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                    scaled_data,
                    yss,
                    distances,
                    label,
                    num_features,
                    model_regressor=model_regressor,
                    feature_selection=self.feature_selection)

        if self.mode == "regression":
            ret_exp.intercept[1] = ret_exp.intercept[0]
            ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
            ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]

        import pandas as pd
        self.lime_preds  = yss

        return ret_exp

    #Override __data_inverse to return also Lime generated points
    def __data_inverse(self,
                       data_row,
                       num_samples):
        """Generates a neighborhood around a prediction.

        For numerical features, perturb them by sampling from a Normal(0,1) and
        doing the inverse operation of mean-centering and scaling, according to
        the means and stds in the training data. For categorical features,
        perturb by sampling according to the training distribution, and making
        a binary feature that is 1 when the value is the same as the instance
        being explained.

        Args:
            data_row: 1d numpy array, corresponding to a row
            num_samples: size of the neighborhood to learn the linear model

        Returns:
            A tuple (data, inverse), where:
                data: dense num_samples * K matrix, where categorical features
                are encoded with either 0 (not equal to the corresponding value
                in data_row) or 1. The first row is the original instance.
                inverse: same as data, except the categorical features are not
                binary, but categorical (as the original data)
        """
        is_sparse = sp.sparse.issparse(data_row)
        if is_sparse:
            num_cols = data_row.shape[1]
            data = sp.sparse.csr_matrix((num_samples, num_cols), dtype=data_row.dtype)
        else:
            num_cols = data_row.shape[0]
            data = np.zeros((num_samples, num_cols))
        categorical_features = range(num_cols)
        if self.discretizer is None:
            instance_sample = data_row
            scale = self.scaler.scale_
            mean = self.scaler.mean_
            if is_sparse:
                # Perturb only the non-zero values
                non_zero_indexes = data_row.nonzero()[1]
                num_cols = len(non_zero_indexes)
                instance_sample = data_row[:, non_zero_indexes]
                scale = scale[non_zero_indexes]
                mean = mean[non_zero_indexes]
            data = self.random_state.normal(
                0, 1, num_samples * num_cols).reshape(
                num_samples, num_cols)
            if self.sample_around_instance:
                data = data * scale + instance_sample
            else:
                data = data * scale + mean
            if is_sparse:
                if num_cols == 0:
                    data = sp.sparse.csr_matrix((num_samples,
                                                 data_row.shape[1]),
                                                dtype=data_row.dtype)
                else:
                    indexes = np.tile(non_zero_indexes, num_samples)
                    indptr = np.array(
                        range(0, len(non_zero_indexes) * (num_samples + 1),
                              len(non_zero_indexes)))
                    data_1d_shape = data.shape[0] * data.shape[1]
                    data_1d = data.reshape(data_1d_shape)
                    data = sp.sparse.csr_matrix(
                        (data_1d, indexes, indptr),
                        shape=(num_samples, data_row.shape[1]))
            categorical_features = self.categorical_features
            first_row = data_row
        else:
            first_row = self.discretizer.discretize(data_row)
        data[0] = data_row.copy()
        inverse = data.copy()
        for column in categorical_features:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = self.random_state.choice(values, size=num_samples,
                                                      replace=True, p=freqs)
            binary_column = (inverse_column == first_row[column]).astype(int)
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        if self.discretizer is not None:
            inverse[1:] = self.discretizer.undiscretize(inverse[1:])
        inverse[0] = data_row
        self.data = data
        return data, inverse



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
        try:
            stdevs_beta = compute_WLS_stdevs(X=self.base.X, Y=self.base.true_labels,
                                             weights=self.base.weights, alpha=self.base.alpha)
        except LinAlgError:
            if np.quantile(self.base.weights,2/len(self.base.weights)) == 0:
                """Kernel width too small, only the chosen unit has meaningful weight."
                      This causes the covariance matrix to be singular"""
            return None

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
        Rsquared = []
        intercept = []
        for i in range(n_calls):
            exp = self.explain_instance(data_row,
                                  predict_fn,
                                  labels,
                                  top_labels,
                                  num_features,
                                  num_samples,
                                  distance_metric,
                                  model_regressor)
            Rsquared.append(exp.score)
            intercept.append(exp.intercept)


            # The first time check if the local model is WRR, otherwise raise Exception
            if not i:
                if self.base.alpha is None:
                    raise LocalModelError("""Lime Local Model is not a Weighted Ridge Regression (WRR),
                    Stability indices may not be computed: the formula is model specific""")


            ci = self.confidence_intervals()
            if ci:
                confidence_intervals.append(ci)

        # Set back the original verbosity
        self.base.verbose = previous_verbose

        # check of the number of variables in the explanations (lasso may retrieve a different number)
        if len(confidence_intervals) < 2:
            print("""Cannot Calculate the indices: kernel width and Ridge penalty was to small.
            No sampled unit got meaningful weight in more than one repetition.""")
            return [np.NaN, np.NaN, len(confidence_intervals)]

        if len(confidence_intervals) != n_calls:
            print("""Some repetitions were not meaningful (no unit had meaningful weight).
            Indices were calculated on {} repetitions""".format(len(confidence_intervals)))

        features_in_exp = [len(conf_int) for conf_int in confidence_intervals]
        if int(np.median(features_in_exp)) != int(np.min(features_in_exp)) != int(np.max(features_in_exp)):
            print("""Pay Attention! Lime Variable Selection selected a different number of variables for the explanations!
                  Stability Indices are not guaranteed to be accurate""")

        csi, vsi = compare_confints(confidence_intervals=confidence_intervals,
                                index_verbose=index_verbose)

        return csi, vsi, len(confidence_intervals)




    #
    # def mean_explanation(self, confidence_intervals,n_calls,data_row):
    #     """Return the mean explanation of the n_calls of LIME"""
    #
    #     features_limes = []
    #     for conf_int in confidence_intervals:
    #         features_limes.append(conf_int.keys())
    #     unique_features = list(set([l for ll in features_limes for l in ll]))
    #
    #     mean_exp = {}
    #     for feat in unique_features:
    #         conf_int_feat = []
    #         for conf_int in confidence_intervals:
    #             conf_int_feat.append(conf_int.get(feat))
    #         mean_exp[feat] = conf_int_feat
    #
    #     final_features = []
    #     for k in sorted(mean_exp, key=lambda k: len(d[k]), reverse=True):
    #         final_features.append(k)
    #     final_features = final_features[:n_calls]
    #     final_exp = {}
    #     for feat in final_features:
    #         conf_ints = mean_exp[feat]
    #         coeff = []
    #         for conf_int in conf_ints:
    #             coeff.append(mean(conf_int))
    #         final_exp[feat] = mean(coeff)
    #
    #     for _ in range(n_calls):
    #         feat = max(mean_exp.items(), key=operator.itemgetter(1))[0]
    #         final_features.append(feat)
    #
    #     mean_explanation = TableDomainMapper(list(final_exp.keys()),
    #                                          list(final_exp.values()),
    #                                          scaled_row=data_row,
    #                                          categorical_features=self.categorical_features,
    #                                          )
    #
    #     mean_explanation.


from sklearn.base import BaseEstimator, RegressorMixin



class Interval(object):
    """Classe per testare  se un valore è dentro a un intervallo
    preso da https://stackoverflow.com/questions/30255450/how-to-check-if-a-number-is-in-a-interval"""
    def __init__(self, middle, deviation):
        self.lower = middle - abs(deviation)
        self.upper = middle + abs(deviation)

    def __contains__(self, item):
        return self.lower <= item <= self.upper

def interval(middle, deviation):
    return Interval(middle, deviation)

class Sklearn_Lime(BaseEstimator, RegressorMixin):

    def __init__(self,
                 epsilon = 0.001,
                 maxRsquared=1.0,
                 penalty = 1,
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
                 training_data_stats=None,
                 labels=(1,),
                 top_labels=None,
                 num_features=10,
                 num_samples=5000,
                 distance_metric='euclidean',
                 model_regressor=None):

        self.maxRsquared =  maxRsquared
        self.penalty = penalty
        self.epsilon = epsilon
        self.mode = mode
        self.training_labels = training_labels
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.categorical_names = categorical_names
        self.kernel_width = kernel_width
        self.kernel = kernel
        self.verbose = verbose
        self.class_names = class_names
        self.feature_selection = feature_selection
        self.discretize_continuous = discretize_continuous
        self.discretizer = discretizer
        self.sample_around_instance = sample_around_instance
        self.random_state = random_state
        self.training_data_stats = training_data_stats
        self.labels = labels
        self.top_labels = top_labels
        self.num_features = num_features
        self.num_samples = num_samples
        self.distance_metric = distance_metric
        self.model_regressor = model_regressor

    def fit(self, X, y=None):

        self.my_lime = LimeTabularExplainerOvr(training_data=X,
                                               penalty=self.penalty,
                                               mode=self.mode,
                                               training_labels=self.training_labels,
                                               feature_names=self.feature_names,
                                               categorical_features=self.categorical_features,
                                               categorical_names=self.categorical_names,
                                               kernel_width=self.kernel_width,
                                               kernel=self.kernel,
                                               verbose=self.verbose,
                                               class_names=self.class_names,
                                               feature_selection=self.feature_selection,
                                               discretize_continuous=self.discretize_continuous,
                                               discretizer=self.discretizer,
                                               sample_around_instance=self.sample_around_instance,
                                               random_state=self.random_state,
                                               training_data_stats=self.training_data_stats)
        return self

    def _get_lime_exp(self, data_row, predict_fn):

        self.explanation = self.my_lime.explain_instance(
            data_row,
            predict_fn,
            labels=self.labels,
            top_labels=self.top_labels,
            num_features=self.num_features,
            num_samples=self.num_samples,
            distance_metric=self.distance_metric,
            model_regressor=self.model_regressor)

        return self.explanation

    def predict(self, X, predict_function, y=None):

        try:
            getattr(self, "my_lime")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        my_exp = self._get_lime_exp(data_row=X, predict_fn=predict_function)

        return my_exp

    def score(self, X, predict_function, y=None, sample_weight=None):

        exp = self.predict(X,predict_function)

        if self.epsilon:
            if exp.intercept[0] in interval(float(exp.local_pred),self.epsilon) and exp.score in interval(1,10*self.epsilon):
                Rquadro = 0
        elif exp.score > self.maxRsquared:
            Rquadro = self.maxRsquared - (exp.score-self.maxRsquared)
        else:
            Rquadro = exp.score

        return Rquadro
