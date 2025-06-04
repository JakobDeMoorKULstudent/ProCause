from abc import ABCMeta, abstractmethod
from numbers import Number
import numpy as np
import pandas as pd
import torch
from scipy import stats
from Seq.utils import T, Y, to_np_vectors, to_np_vector, to_torch_variable, permutation_test, regular_round
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, average_precision_score, accuracy_score
from copy import deepcopy

MODEL_LABEL = "model"
TRUE_LABEL = "true"
T_MODEL_LABEL = "{} ({})".format(T, MODEL_LABEL)
Y_MODEL_LABEL = "{} ({})".format(Y, MODEL_LABEL)
T_TRUE_LABEL = "{} ({})".format(T, TRUE_LABEL)
Y_TRUE_LABEL = "{} ({})".format(Y, TRUE_LABEL)
SEED = 42
TRAIN = "train"
VAL = "val"
TEST = "test"


class BaseGenModelMeta(ABCMeta):
    """
    Forces subclasses to implement abstract_attributes
    """

    abstract_attributes = []

    def __call__(cls, *args, **kwargs):
        obj = super(BaseGenModelMeta, cls).__call__(*args, **kwargs)
        missing_attributes = []
        for attr_name in obj.abstract_attributes:
            if not hasattr(obj, attr_name):
                missing_attributes.append(attr_name)
        if len(missing_attributes) == 1:
            raise TypeError(
                "Can't instantiate abstract class {} with abstract attribute '{}'. "
                "You must set self.{} in the constructor.".format(
                    cls.__name__, missing_attributes[0], missing_attributes[0]
                )
            )
        elif len(missing_attributes) > 1:
            raise TypeError(
                "Can't instantiate abstract class {} with abstract attributes {}. "
                "For example, you must set self.{} in the constructor.".format(
                    cls.__name__, missing_attributes, missing_attributes[0]
                )
            )

        return obj


class BaseGenModel(object, metaclass=BaseGenModelMeta):
    """
    Abstract class for generative models. Implementations of 2 methods and
    3 attributes are required.

    2 methods:
        sample_t(w) - models p(t | w)
        sample_y(t, w) - models p(y | t, w)

    3 attributes:
        w - covariates from real data
        t - treatments from real data
        y - outcomes from real data
    """

    abstract_attributes = ["data_train", "data_val", "data_test", "prep_utils", "MODEL_PARAMS"]

    def __init__(self, data_train, data_val, data_test, prep_utils, MODEL_PARAMS):
        
        """
        Initialize the generative model. Split the data up according to the
        splits specified by train_prop, val_prop, and test_prop. These can add
        to 1, or they can just be arbitary numbers that correspond to the
        unnormalized fraction of the dataset for each subsample.
        :param w: ndarray of covariates
        :param t: ndarray for vector of treatment
        :param y: ndarray for vector of outcome
        :param train_prop: number to use for proportion of the whole dataset
            that is in the training set
        :param val_prop: number to use for proportion of the whole dataset that
            is in the validation set
        :param test_prop: number to use for proportion of the whole dataset that
            is in the test set
        :param test_size: size of the test set
        :param shuffle: boolean for whether to shuffle the data
        :param seed: random seed for pytorch and numpy
        :param w_transform: transform for covariates
        :param t_transform: transform for treatment
        :param y_transform: transform for outcome
        :param verbose: boolean
        """

        self.data_train = data_train
        self.data_val = data_val
        self.data_test = data_test
        self.prep_utils = prep_utils
        self.MODEL_PARAMS = MODEL_PARAMS
        self.set_seed(self.MODEL_PARAMS["seed"])

    def get_data(self, dataset=TRAIN, verbose=True):
        """
        Get the specific dataset. Splits were determined in the constructor.

        :param transformed: If True, use transformed version of data.
            If False, use original (non-transformed) version of data.
        :param dataset: dataset subset to use (train, val, or test)
        :param verbose:
        :return: (covariates, treatment, outcome)
        """
        dataset = dataset.lower()
        if dataset == TRAIN:
            x_case, x_event, prefix_len, t, y = self.data_train["X_case"], self.data_train["X_event"], self.data_train["prefix_len"], self.data_train["T"], self.data_train["Y"]
        elif dataset == VAL or dataset == "validation":
            x_case, x_event, prefix_len, t, y = self.data_val["X_case"], self.data_val["X_event"], self.data_val["prefix_len"], self.data_val["T"], self.data_val["Y"]
        elif dataset == TEST:
            x_case, x_event, prefix_len, t, y = self.data_test["X_case"], self.data_test["X_event"], self.data_test["prefix_len"], self.data_test["T"], self.data_test["Y"]
        else:
            raise ValueError("Invalid dataset: {}".format(dataset))

        t = t.to(torch.float32)
        return x_case, x_event, prefix_len, t, y

    def sample_x(self, dataset=TRAIN):
        if dataset == TEST:
            return self.data_test["X_case"], self.data_test["X_event"], self.data_test["prefix_len"]
        elif dataset == VAL:
            return self.data_val["X_case"], self.data_val["X_event"], self.data_val["prefix_len"]
        else:
            return self.data_train["X_case"], self.data_train["X_event"], self.data_train["prefix_len"]

    @abstractmethod
    def _sample_t(self, w, overlap=1):
        pass

    @abstractmethod
    def _sample_y(self, t, w, ret_counterfactuals=False):
        pass

    @abstractmethod
    def mean_y(self, t, w):
        pass

    def sample_t(self, x_case, x_event, prefix_len=None, overlap=1):
        """
        Sample the treatment vector.

        :param w: covariate (confounder)
        :param untransform: whether to transform the data back to the raw scale
        :param overlap: if 1, leave treatment untouched;
            if 0, push p(T = 1 | w) to 0 for all w where p(T = 1 | w) < 0.5 and
            and push p(T = 1 | w) to 1 for all w where p(T = 1 | w) >= 0.5
            if 0 < overlap < 1, do a linear interpolation of the above
        :param seed: random seed
        :return: sampled treatment
        """
        if x_case is None or x_event is None or prefix_len is None:
            x_case, x_event, prefix_len = self.sample_x()
        t = self._sample_t(x_case, x_event, prefix_len=prefix_len, overlap=overlap)
        return t

    def sample_y(self, x_case, x_event, t, prefix_len=None, causal_effect_scale=None,
                 deg_hetero=1.0, ret_counterfactuals=False, seed=None):
        """
        :param t: treatment
        :param w: covariate (confounder)
        :param untransform: whether to transform the data back to the raw scale
        :param causal_effect_scale: scale of the causal effect (size of ATE)
        :param deg_hetero: degree of heterogeneity (between 0 and 1)
            When deg_hetero=1, y1 and y0 remain unchanged. When deg_hetero=0,
            y1 - y0 is the same for all individuals.
        :param ret_counterfactuals: return counterfactuals if True
        :param seed: random seed
        :return: sampled outcome
        """
        if seed is not None:
            self.set_seed(seed)
        if x_case is None or x_event is None or prefix_len is None:
            x_case, x_event, prefix_len = self.sample_x()
        y_total = self._sample_y(t, x_case, x_event, prefix_len=prefix_len, ret_counterfactuals=True)
        # y0, y1 = self._sample_y(t, x_case, x_event, prefix_len=prefix_len, ret_counterfactuals=True)

        if deg_hetero == 1.0 and causal_effect_scale == None:  # don't change heterogeneity or causal effect size
            pass
        else:   # change degree of heterogeneity and/or causal effect size
            # degree of heterogeneity
            if deg_hetero != 1.0:
                assert 0 <= deg_hetero < 1, f'deg_hetero not in [0, 1], got {deg_hetero}'
                y1_mean = y1.mean()
                y0_mean = y0.mean()
                ate = y1_mean - y0_mean

                # calculate value to shrink either y1 or y0 (whichever is
                # further from its mean) to when deg_hetero = 0
                further_y1 = np.greater(np.abs(y1 - y1_mean), np.abs(y0 - y0_mean))
                further_y0 = np.logical_not(further_y1)
                alpha = np.random.rand(len(y1))
                y1_limit = further_y1 * ((1 - alpha) * y1 + alpha * y1_mean)
                y0_limit = further_y0 * ((1 - alpha) * y0 + alpha * y0_mean)

                # shrink y1 (or y0) and calculate corresponding y0 or (y1) based on
                scaled_y1 = (1 - deg_hetero) * y1_limit + deg_hetero * y1 * further_y1
                corresponding_y0 = (1 - deg_hetero) * (scaled_y1 - ate) + deg_hetero * y0 * further_y1
                scaled_y0 = (1 - deg_hetero) * y0_limit + deg_hetero * y0 * further_y0
                corresponding_y1 = (1 - deg_hetero) * (scaled_y0 + ate) + deg_hetero * y1 * further_y0
                y1 = scaled_y1 * further_y1 + corresponding_y1 * further_y0
                y0 = scaled_y0 * further_y0 + corresponding_y0 * further_y1

            # size of causal effect
            if causal_effect_scale is not None:
                ate = (y1 - y0).mean()
                y1 = causal_effect_scale / ate * y1
                y0 = causal_effect_scale / ate * y0

        if ret_counterfactuals:
            # return y0, y1
            return y_total
        else:
            # return y0 * (1 - t) + y1 * t
            y_ = torch.stack(y_total, dim=1)
            indices = torch.argmax(t.int(), dim=1) if self.dims["dim_t"] > 1 else t.int().squeeze(-1)
            # indices = torch.argmax(t, dim=1) if self.dims["dim_t"] > 1 else t.squeeze(-1)
            y_ = y_[torch.arange(y_.size(0)), indices.long() if indices is torch.Tensor else indices]
            return y_

    def set_seed(self, seed=SEED):
        torch.manual_seed(seed)
        np.random.seed(seed)

    def sample(self, x_case=None, x_event=None, prefix_len=None, dataset=TRAIN, overlap=1,
               causal_effect_scale=None, deg_hetero=1.0, ret_counterfactuals=False, seed=None):
        """
        Sample from generative model.

        :param w: covariates (confounders)
        :param transform_w: whether to transform the w (if given)
        :param untransform: whether to transform the data back to the raw scale
        :param seed: random seed
        :param dataset: train or test for sampling w from
        :param overlap: if 1, leave treatment untouched;
            if 0, push p(T = 1 | w) to 0 for all w where p(T = 1 | w) < 0.5 and
            and push p(T = 1 | w) to 1 for all w where p(T = 1 | w) >= 0.5
            if 0 < overlap < 1, do a linear interpolation of the above
        :param causal_effect_scale: scale of the causal effect (size of ATE)
        :param deg_hetero: degree of heterogeneity (between 0 and 1)
            When deg_hetero=1, y1 and y0 remain unchanged. When deg_hetero=0,
            y1 - y0 is the same for all individuals.
        :param ret_counterfactuals: return counterfactuals if True
        :return: (w, t, y)
        """
        if x_case is None or x_event is None or prefix_len is None:
            x_case, x_event, prefix_len = self.sample_x(dataset=dataset)
        t = self.sample_t(x_case, x_event, prefix_len=prefix_len, overlap=overlap)
        if ret_counterfactuals:
            y_total = self.sample_y(x_case, x_event, t, prefix_len=prefix_len, causal_effect_scale=causal_effect_scale,
                                    deg_hetero=deg_hetero, ret_counterfactuals=True, seed=seed)
            return x_case, x_event, prefix_len, t, y_total
            # y0, y1 = self.sample_y(x_case, x_event,
            #     t, prefix_len=prefix_len, causal_effect_scale=causal_effect_scale,
            #     deg_hetero=deg_hetero, ret_counterfactuals=True
            # )
            # return x_case, x_event, prefix_len, t, y0, y1
        else:
            y = self.sample_y(x_case, x_event, t, prefix_len=prefix_len, causal_effect_scale=causal_effect_scale,
                              deg_hetero=deg_hetero, ret_counterfactuals=False, seed=seed)
            return x_case, x_event, prefix_len, t, y

    def sample_interventional(self, t, w=None, seed=None, causal_effect_scale=None, deg_hetero=1.0):
        if seed is not None:
            self.set_seed(seed)
        if w is None:
            w = self.sample_w(untransform=False)
        if isinstance(w, Number):
            raise ValueError('Unsupported data type: {} ... only numpy is currently supported'.format(type(w)))
        if isinstance(t, Number):
            t = np.full_like(self.t, t)
        return self.sample_y(t, w, causal_effect_scale=causal_effect_scale, deg_hetero=deg_hetero)

    def ate(self, t1=1, t0=0, w=None, noisy=True, untransform=True, transform_t=True, n_y_per_w=100,
            causal_effect_scale=None, deg_hetero=1.0):
        return self.ite(t1=t1, t0=t0, w=w, noisy=noisy, untransform=untransform,
                        transform_t=transform_t, n_y_per_w=n_y_per_w,
                        causal_effect_scale=causal_effect_scale,
                        deg_hetero=deg_hetero).mean()

    def noisy_ate(self, t1=1, t0=0, w=None, n_y_per_w=100, seed=None, transform_w=False):
        if w is not None and transform_w:
            w = self.w_transform.transform(w)

        # Note: bad things happen if w is not transformed and transform_w is False

        if seed is not None:
            self.set_seed(seed)

        if (isinstance(t1, Number) or isinstance(t0, Number)) and w is not None:
            t_shape = list(self.t.shape)
            t_shape[0] = w.shape[0]
            t1 = np.full(t_shape, t1)
            t0 = np.full(t_shape, t0)
        total = 0
        for _ in range(n_y_per_w):
            total += (self.sample_interventional(t=t1, w=w) -
                      self.sample_interventional(t=t0, w=w)).mean()
        return total / n_y_per_w

    def att(self, t1=1, t0=0, w=None, untransform=True, transform_t=True):
        pass

    def ite(self, t1=1, t0=0, w=None, t=None, untransform=True, transform_t=True, transform_w=True,
            estimand="all", noisy=True, seed=None, n_y_per_w=100,
            causal_effect_scale=None, deg_hetero=1.0):
        if seed is not None:
            self.set_seed(seed)
        if w is None:
            # w = self.w_transformed
            w = self.sample_w(untransform=False)
            t = self.t
        estimand = estimand.lower()
        if estimand == "all" or estimand == "ate":
            pass
        elif estimand == "treated" or estimand == "att":
            w = w[t == 1]
        elif estimand == "control" or estimand == "atc":
            w = w[t == 0]
        else:
            raise ValueError("Invalid estimand: {}".format(estimand))
        if transform_t:
            t1 = self.t_transform.transform(t1)
            t0 = self.t_transform.transform(t0)
            # Note: check that this is an identity transformation
        if isinstance(t1, Number) or isinstance(t0, Number):
            t_shape = list(self.t.shape)
            t_shape[0] = w.shape[0]
            t1 = np.full(t_shape, t1)
            t0 = np.full(t_shape, t0)
        if noisy:
            y1_total = np.zeros(w.shape[0])
            y0_total = np.zeros(w.shape[0])
            for _ in range(n_y_per_w):
                y1_total += to_np_vector(self.sample_interventional(
                    t=t1, w=w,causal_effect_scale=causal_effect_scale, deg_hetero=deg_hetero))
                y0_total += to_np_vector(self.sample_interventional(
                    t=t0, w=w, causal_effect_scale=causal_effect_scale, deg_hetero=deg_hetero))
            y_1 = y1_total / n_y_per_w
            y_0 = y0_total / n_y_per_w
        else:
            if causal_effect_scale is not None or deg_hetero != 1.0:
                raise ValueError('Invalid causal_effect_scale or deg_hetero. '
                                 'Current mean_y only supports defaults.')
            y_1 = to_np_vector(self.mean_y(t=t1, w=w))
            y_0 = to_np_vector(self.mean_y(t=t0, w=w))
        return y_1 - y_0

    # def plot_ty_dists(self, joint=True, marginal_hist=True, marginal_qq=True,
    #                   dataset=TRAIN, verbose=True,
    #                   title=True, name=None, file_ext='pdf', thin_model=None,
    #                   thin_true=None, joint_kwargs={}, test=False):
    #     """
    #     Creates up to 3 different plots of the real data and the corresponding model

    #     :param joint: boolean for whether to plot p(t, y)
    #     :param marginal_hist: boolean for whether to plot the p(t) and p(y) histograms
    #     :param marginal_qq: boolean for whether to plot the p(y) Q-Q plot
    #         or use 'both' for plotting both the p(t) and p(y) Q-Q plots
    #     :param dataset: dataset subset to use (train, val, or test)
    #     :param transformed: If True, use transformed version of data.
    #         If False, use original (non-transformed) version of data.
    #     :param title: boolean for whether or not to include title in plots
    #     :param name: name to use in plot titles and saved files defaults to name of class
    #     :param file_ext: file extension to for saving plots (e.g. 'pdf', 'png', etc.)
    #     :param thin_model: thinning interval for the model data
    #     :param thin_true: thinning interval for the real data
    #     :param joint_kwargs: kwargs passed to sns.kdeplot() for p(t, y)
    #     :param test: if True, does not show or save plots
    #     :param seed: seed for sample from generative model
    #     :return:
    #     """
    #     if name is None:
    #         name = self.__class__.__name__

    #     _, _, _, t_model, y_model = to_np_vectors(self.sample(dataset=dataset),
    #                                         thin_interval=thin_model)
    #     _, _, _, t_true, y_true = self.get_data(dataset=dataset, verbose=verbose)
    #     t_true, y_true = to_np_vectors((t_true, y_true), thin_interval=thin_true)
    #     plots = []

    #     if joint:
    #         fig1 = compare_joints(t_model, y_model, t_true, y_true,
    #                        xlabel1=T_MODEL_LABEL, ylabel1=Y_MODEL_LABEL,
    #                        xlabel2=T_TRUE_LABEL, ylabel2=Y_TRUE_LABEL,
    #                        xlabel=T, ylabel=Y,
    #                        label1=MODEL_LABEL, label2=TRUE_LABEL,
    #                        save_fname='{}_ty_joints.{}'.format(name, file_ext),
    #                        title=title, name=name, test=test, kwargs=joint_kwargs)
            
    #         plots += [fig1]

    #     if marginal_hist or marginal_qq:
    #         fig2 = compare_bivariate_marginals(t_model, t_true, y_model, y_true,
    #                                     xlabel=T, ylabel=Y,
    #                                     label1=MODEL_LABEL, label2=TRUE_LABEL,
    #                                     # hist=marginal_hist, qqplot=marginal_qq,
    #                                     hist=marginal_hist, qqplot=False,
    #                                     save_hist_fname='{}_ty_marginal_hists.{}'.format(name, file_ext),
    #                                     save_qq_fname='{}_ty_marginal_qqplots.{}'.format(name, file_ext),
    #                                     title=title, name=name, test=test)
    #         plots += [fig2]
    #     return plots

    def get_univariate_quant_metrics(self, dataset=TRAIN, verbose=True,
                                     thin_model=None, thin_true=None, n=None, outcome_distribution=None, t_only=False, y_only=False,
                                     t_model=None, y_model=None, seed=None):
        """
        Calculates quantitative metrics for the difference between p(t) and
        p_model(t) and the difference between p(y) and p_model(y)

        :param dataset: dataset subset to evaluate on (train, val, or test)
        :param transformed: If True, use transformed version of data.
            If False, use original (non-transformed) version of data.
        :param thin_model: thinning interval for the model data
        :param thin_true: thinning interval for the real data
        :param seed: seed for sample from generative model
        :return: {
            't_ks_pval': ks p-value with null that t_model and t_true are from the same distribution
            'y_ks_pval': ks p-value with null that y_model and y_true are from the same distribution
            't_wasserstein1_dist': wasserstein1 distance between t_true and t_model
            'y_wasserstein1_dist': wasserstein1 distance between y_true and y_model
        }
        """
        # print("t_model.shape at the start of get_univariate_quant_metrics", t_model.shape)
        # print("y_model.shape at the start of get_univariate_quant_metrics", y_model.shape)
        if t_model is None or y_model is None:
            _, _, _, t_model, y_model = self.sample(dataset=dataset)
        t_model = torch.argmax(t_model, dim=1) if self.dims["dim_t"] > 1 else t_model.squeeze(-1)
        t_model, y_model = to_np_vectors((t_model, y_model), thin_interval=thin_model)
        # _, _, _, t_model, y_model = to_np_vectors(
        #     self.sample(dataset=dataset),
        #     thin_interval=thin_model
        # )

        _, _, _, t_true, y_true = self.get_data(dataset=dataset, verbose=verbose)
        t_true = torch.argmax(t_true, dim=1) if self.dims["dim_t"] > 1 else t_true.squeeze(-1)
        t_true, y_true = to_np_vectors((t_true, y_true), thin_interval=thin_true)

        # jitter for numerical stability
        # make sure to set seed to ensure reproducibility
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(SEED)
        t_true = t_true.copy() + np.random.rand(*t_true.shape) * 1e-6
        t_model = t_model.copy() + np.random.rand(*t_model.shape) * 1e-6

        # print y_model and y_true if they contain non-finite values
        if not np.all(np.isfinite(y_model)):
            print("y_model contains non-finite values")
            print(y_model)
        if not np.all(np.isfinite(y_true)):
            print("y_true contains non-finite values")
            print(y_true)

        ks_label = "_ks_pval"
        es_label = "_es_pval"
        wasserstein_label = "_wasserstein1_dist"

        y_model = np.round(y_model, 4)
        y_true = np.round(y_true, 4)

        metrics = {}
        if not y_only:
            metrics.update({
                T + ks_label: float(stats.ks_2samp(t_model, t_true).pvalue),
                T + es_label: float(stats.epps_singleton_2samp(t_model, t_true).pvalue),
                T + wasserstein_label: float(stats.wasserstein_distance(t_model, t_true))
            })

        print('t_model.shape', t_model.shape)
        print('t_true.shape', t_true.shape)
        print('y_model.shape', y_model.shape)
        print('y_true.shape', y_true.shape)

        if not t_only:
            if outcome_distribution.__class__.__name__ == "MixedDistributionAtoms":
                atoms_rounded = np.round(outcome_distribution.atoms, 4)
                tolerance = 1e-4
                y_atoms_true = y_true[np.any(np.isclose(y_true[:, None], atoms_rounded, atol=tolerance), axis=1)]
                y_atoms_model = y_model[np.any(np.isclose(y_model[:, None], atoms_rounded, atol=tolerance), axis=1)]
                y_cont_true = np.where(np.isin(y_true, atoms_rounded, invert=True), y_true, np.nan)
                y_cont_true = y_cont_true[~np.isnan(y_cont_true)]
                y_cont_model = np.where(np.isin(y_model, atoms_rounded, invert=True), y_model, np.nan)
                y_cont_model = y_cont_model[~np.isnan(y_cont_model)]

                # make sure none of the special y's have less than 5 elements
                special_ys = [y_atoms_true, y_atoms_model, y_cont_true, y_cont_model]
                # print("special_ys", special_ys)
                # print("special_ys[0]", special_ys[0])
                # print("special_ys[1]", special_ys[1])
                # print("special_ys[2]", special_ys[2])
                # print("special_ys[3]", special_ys[3])
                for index in range(len(special_ys)):
                    while len(special_ys[index]) < 5:
                        if len(special_ys[index]) == 0:
                            # just make a random array
                            special_ys[index] = np.random.rand(5)
                        else:
                            # print("special_y", special_ys[index])
                            # add special_y to itself until it has at least 5 elements
                            # new_special_y = np.concatenate([special_y, special_y])
                            special_ys[index] = np.concatenate([special_ys[index], special_ys[index]])
                            # special_ys[index] = np.concatenate(special_y)
                
                [y_atoms_true, y_atoms_model, y_cont_true, y_cont_model] = special_ys
                
                metrics.update({
                    Y + ks_label: float(stats.ks_2samp(y_model, y_true).pvalue),
                    Y + ks_label + "_atoms": float(stats.ks_2samp(y_atoms_model, y_atoms_true).pvalue),
                    Y + ks_label + "_cont": float(stats.ks_2samp(y_cont_model, y_cont_true).pvalue),
                    Y + es_label: float(stats.epps_singleton_2samp(y_model, y_true).pvalue),
                    Y + es_label + "_atoms": float(stats.epps_singleton_2samp(y_atoms_model, y_atoms_true).pvalue),
                    Y + es_label + "_cont": float(stats.epps_singleton_2samp(y_cont_model, y_cont_true).pvalue),
                    Y + wasserstein_label: float(stats.wasserstein_distance(y_model, y_true)),
                    Y + wasserstein_label + "_atoms": float(stats.wasserstein_distance(y_atoms_model, y_atoms_true)),
                    Y + wasserstein_label + "_cont": float(stats.wasserstein_distance(y_cont_model, y_cont_true))
                })
            else:
                metrics.update({
                    Y + ks_label: float(stats.ks_2samp(y_model, y_true).pvalue),
                    Y + es_label: float(stats.epps_singleton_2samp(y_model, y_true).pvalue),
                    Y + wasserstein_label: float(stats.wasserstein_distance(y_model, y_true))
                })

        print("Univariate Quantitative Metrics")
        print(metrics)

        # make sure to delete the y_model and y_true from memory
        del y_model
        del y_true
        del t_model
        del t_true
        if not t_only:
            if outcome_distribution.__class__.__name__ == "MixedDistributionAtoms":
                del y_atoms_true
                del y_atoms_model
                del y_cont_true
                del y_cont_model

        return metrics

    def get_multivariate_quant_metrics(
        self,
        include_x=True,
        dataset=TRAIN,
        # norm=2,
        norm=1,
        k=1,
        alphas=None,
        n_permutations=1000,
        verbose=False,
        n=None,
        calculate_wssd=True,
        seed=None,
        t_model=None,
        y_model=None,
    ):
        """
        Computes Wasserstein-1 and Wasserstein-2 distances. Also computes all the
        test statistics and p-values for the multivariate two sample tests from
        the torch_two_sample package. See that documentation for more info on
        the specific tests: https://torch-two-sample.readthedocs.io/en/latest/

        :param include_w: If False, test if p(t, y) = p_model(t, y).
            If True, test if p(w, t, y) = p(w, t, y).
        :param dataset: dataset subset to evaluate on (train, val, or test)
        :param transformed: If True, use transformed version of data.
            If False, use original (non-transformed) version of data.
        :param norm: norm used for Friedman-Rafsky test and kNN test
        :param k: number of nearest neighbors to use for kNN test
        :param alphas: list of kernel parameters for MMD test
        :param n_permutations: number of permutations for each test
        :param seed: seed for sample from generative model
        :param verbose: print intermediate steps
        :param n: subsample dataset to n samples

        :return: {
            'wasserstein1_dist': wasserstein1 distance between p_true and p_model
            'wasserstein2_dist': wasserstein2 distance between p_true and p_model
            'Friedman-Rafsky pval': p-value for Friedman-Rafsky test with null
                that p_true and p_model are from the same distribution
            'kNN pval': p-value for kNN test with null that p_true and p_model are from the same distribution
            'MMD pval': p-value for MMD test with null that p_true and p_model are from the same distribution
            'Energy pval': p-value for the energy test with null that p_true and p_model are from the same distribution
        }
        """
        try:
            import ot
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                str(e)
                + " ... Install: conda install cython && conda install -c conda-forge pot"
            )
        try:
            # import torch_two_sample
            from src.utils.torch_two_sample_master.torch_two_sample.statistics_nondiff import FRStatistic, KNNStatistic
            from src.utils.torch_two_sample_master.torch_two_sample.statistics_diff import MMDStatistic, EnergyStatistic
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(str(e) + ' ... Install: pip install git+git://github.com/josipd/torch-two-sample')
        if t_model is None or y_model is None:
            _, _, _, t_model, y_model = self.sample(dataset=dataset)
        x_case_model, x_event_model, prefix_len_model, _, _ = self.sample(dataset=dataset)
        t_model = torch.argmax(t_model, dim=1) if self.dims["dim_t"] > 1 else t_model.squeeze(-1)
        old_y_shape = y_model.shape
        print('t_model.shape: ', t_model.shape)
        print('y_model.shape: ', y_model.shape)
        if n is not None and x_case_model.shape[0] > n:

            select_rows = np.random.choice(x_case_model.shape[0], n, replace=False)
            x_case_model = x_case_model[select_rows, :]
            x_event_model = x_event_model[select_rows, :]

            if len(old_y_shape) > 2:
                select_cols = np.random.choice(old_y_shape[0], n, replace=True)
                t_model = t_model[select_cols, select_rows]
                y_model = y_model[select_cols, select_rows]
            else:
                t_model = t_model[select_rows]
                y_model = y_model[select_rows]

        t_model, y_model = to_np_vectors((t_model, y_model), column_vector=True)
        y_model = np.round(y_model, 4)
        model_samples = np.hstack((t_model, y_model))

        x_case_true, x_event_true, prefix_len_true, t_true, y_true = self.get_data(dataset=dataset, verbose=verbose)
        t_true = torch.argmax(t_true, dim=1) if self.dims["dim_t"] > 1 else t_true.squeeze(-1)
        print('t_true.shape: ', t_true.shape)
        print('y_true.shape: ', y_true.shape)
        if n is not None and x_case_true.shape[0] > n:
            x_case_true = x_case_true[select_rows, :]
            x_event_true = x_event_true[select_rows, :]
            t_true = t_true[select_rows]
            y_true = y_true[select_rows]

        t_true, y_true = to_np_vectors((t_true, y_true), column_vector=True)
        y_true = np.round(y_true, 4)
        true_samples = np.hstack((t_true, y_true))

        print("model_samples.shape: ", model_samples.shape)
        print("true_samples.shape: ", true_samples.shape)

        if include_x:
            # flatten x_event
            x_event_model = x_event_model.reshape(x_event_model.shape[0], -1)
            x_event_true = x_event_true.reshape(x_event_true.shape[0], -1)

            model_samples = np.hstack((x_case_model, x_event_model, model_samples))
            true_samples = np.hstack((x_case_true, x_event_true, true_samples))

        n_model = model_samples.shape[0]
        n_true = true_samples.shape[0]

        a, b = np.ones((n_model,)) / n_model, np.ones((n_true,)) / n_true  # uniform   # uniform distribution on samples

        results = {}

        if calculate_wssd:
            def calculate_wasserstein1_dist(x, y):
                M_wasserstein1 = ot.dist(x, y, metric="euclidean")
                wasserstein1_dist = ot.emd2(a, b, M_wasserstein1)
                return wasserstein1_dist
                # pass

            def calculate_wasserstein2_dist(x, y):
                M_wasserstein2 = ot.dist(x, y, metric="sqeuclidean")
                wasserstein2_dist = np.sqrt(ot.emd2(a, b, M_wasserstein2))
                return wasserstein2_dist
                pass

            wasserstein1_pval = permutation_test(model_samples, true_samples,
                                                func=calculate_wasserstein1_dist,
                                                method='approximate',
                                                num_rounds=n_permutations,
                                                seed=0)
            print("wasserstein1_pval: ", wasserstein1_pval)
            wasserstein2_pval = permutation_test(model_samples, true_samples,
                                                func=calculate_wasserstein2_dist,
                                                method='approximate',
                                                num_rounds=n_permutations,
                                                seed=0)
            print("wasserstein2_pval: ", wasserstein2_pval)

            results = {
                "wasserstein1 pval": wasserstein1_pval,
                "wasserstein2 pval": wasserstein2_pval,
            }

        model_samples_var = to_torch_variable(model_samples)
        true_samples_var = to_torch_variable(true_samples)

        # print("model_samples_var: ", model_samples_var)
        # print("true_samples_var: ", true_samples_var)

        fr = FRStatistic(n_model, n_true)
        matrix = fr(model_samples_var, true_samples_var, norm=norm, ret_matrix=True)[1]
        results["Friedman-Rafsky pval"] = fr.pval(matrix, n_permutations=n_permutations)
        print("Friedman-Rafsky pval: ", results["Friedman-Rafsky pval"])

        knn = KNNStatistic(n_model, n_true, k)
        matrix = knn(model_samples_var, true_samples_var, norm=norm, ret_matrix=True)[1]
        results["kNN pval"] = knn.pval(matrix, n_permutations=n_permutations)
        print("kNN pval: ", results["kNN pval"])

        if alphas is not None:
            mmd = MMDStatistic(n_model, n_true)
            matrix = mmd(model_samples_var, true_samples_var, alphas=None, ret_matrix=True)[1]
            results['MMD pval'] = mmd.pval(matrix, n_permutations=n_permutations)
            print("MMD pval: ", results['MMD pval'])

        energy = EnergyStatistic(n_model, n_true)
        matrix = energy(model_samples_var, true_samples_var, ret_matrix=True)[1]
        results["Energy pval"] = energy.pval(matrix, n_permutations=n_permutations)
        print("Energy pval: ", results["Energy pval"])

        # make sure to delete the model_samples and true_samples from memory
        del model_samples
        del true_samples
        del model_samples_var
        del true_samples_var
        del y_model
        del y_true
        del t_model
        del t_true
        del x_case_model
        del x_event_model
        del prefix_len_model
        del x_case_true
        del x_event_true
        del prefix_len_true

        return results

    def sample_bpic(self, nr_samples_per_case=50, seed=82*82):
        data_prepped = self.data_test

        # NOTE: only used for BPIC logs (so only binary treatment)
        total_outcome_df = pd.DataFrame()
        for sample in range(nr_samples_per_case):
            new_seed = seed + sample*5
            # case_nr in data_prepped is a tensor, so convert to numpy
            case_nrs = data_prepped["case_nr"].numpy()
            outcome_df = pd.DataFrame({"case_nr": case_nrs, "outcome0": [None] * len(data_prepped["case_nr"]), "outcome1": [None] * len(data_prepped["case_nr"]), "t_est": [None] * len(data_prepped["case_nr"])})
            X_case = data_prepped["X_case"]
            X_event = data_prepped["X_event"]
            prefix_len = data_prepped["prefix_len"]
            T = data_prepped["T"]
            _, _, prefix_len, t_est, y_total_ = self.sample(x_case=X_case, x_event=X_event, prefix_len=prefix_len, ret_counterfactuals=True, seed=new_seed)
            y_pred_total_ = np.concatenate(y_total_, axis=1)
            outcome_df["outcome0"] = y_pred_total_[:, 0]
            outcome_df["outcome1"] = y_pred_total_[:, 1]
            outcome_df["t_est"] = t_est
            outcome_df["prefix_len"] = prefix_len
            total_outcome_df = pd.concat([total_outcome_df, outcome_df], axis=0, ignore_index=True)
        return total_outcome_df

    def calculate_metrics_bpic(self, estimated_outcome_df_list, ensemble=False):
        # REFERENCE: TEINEMAA
        data_prepped = self.data_test
        # calculate the F1, precision, recall, roc-auc, pr-auc, comparing the true outcome with the estimated outcomes (nr_samples_per_case=50) per case
        # so match the true outcomes in self.data_test with the estimated outcomes in estimated_outcome_df by using case_nr

        if ensemble:
            # ensemble the estimated_outcome_df_list
            estimated_outcome_df = deepcopy(estimated_outcome_df_list[0])

            t_est_matrix = np.column_stack([df["t_est"].values for df in estimated_outcome_df_list])
            outcome0_matrix = np.column_stack([df["outcome0"].values for df in estimated_outcome_df_list])
            outcome1_matrix = np.column_stack([df["outcome1"].values for df in estimated_outcome_df_list])

            # Compute the mode using majority voting
            n = len(estimated_outcome_df_list)
            estimated_outcome_df["t_est"] = (t_est_matrix.sum(axis=1) > n / 2).astype(int)
            estimated_outcome_df["outcome0"] = (outcome0_matrix.sum(axis=1) > n / 2).astype(int)
            estimated_outcome_df["outcome1"] = (outcome1_matrix.sum(axis=1) > n / 2).astype(int)
        else:
            estimated_outcome_df = estimated_outcome_df_list

        true_outcome_df = pd.DataFrame({"case_nr": data_prepped["case_nr"].numpy(), "y_true": data_prepped["Y"].numpy(), "t_true": data_prepped["T"].flatten().numpy().astype(int), "prefix_len": data_prepped["prefix_len"].numpy()})
        # merge the true_outcome_df with estimated_outcome_df on case_nr and prefix_len
        estimated_outcome_df['y_est'] = estimated_outcome_df.apply(lambda row: row['outcome0'] if row['t_est'] == 0 else row['outcome1'], axis=1)
        merged_df = pd.merge(true_outcome_df, estimated_outcome_df, on=["case_nr", "prefix_len"], suffixes=('_true', '_est'))

        metrics = {"acc_y": {}, "f1_y": {}, "precision_y": {}, "recall_y": {}, "roc_auc_y": {}, "pr_auc_y": {}, "n_cases": {}}
        # calculate the y metrics for each prefix length group, and then take weighted average over the groups, with weights being the number of cases in each group
        for prefix_len in merged_df["prefix_len"].unique():
            prefix_len_group = merged_df[merged_df["prefix_len"] == prefix_len]
            metrics["acc_y"][prefix_len] = accuracy_score(prefix_len_group["y_true"], prefix_len_group["y_est"])
            metrics["f1_y"][prefix_len] = f1_score(prefix_len_group["y_true"], prefix_len_group["y_est"])
            metrics["precision_y"][prefix_len] = precision_score(prefix_len_group["y_true"], prefix_len_group["y_est"])
            metrics["recall_y"][prefix_len] = recall_score(prefix_len_group["y_true"], prefix_len_group["y_est"])
            metrics["pr_auc_y"][prefix_len] = average_precision_score(prefix_len_group["y_true"], prefix_len_group["y_est"])

            # check if the roc_auc_score can be calculated (needs at least 2 classes)
            if len(np.unique(prefix_len_group["y_true"])) > 1 and len(np.unique(prefix_len_group["y_est"])) > 1:
                metrics["roc_auc_y"][prefix_len] = roc_auc_score(prefix_len_group["y_true"], prefix_len_group["y_est"])
            else:
                metrics["roc_auc_y"][prefix_len] = 0.5
            
            metrics["n_cases"][prefix_len] = len(prefix_len_group)

        # calculate the weighted average over the groups
        n_cases = sum(metrics["n_cases"].values())
        for metric in metrics.keys():
            if metric != "n_cases":
                metrics[metric] = sum([metrics[metric][prefix_len] * metrics["n_cases"][prefix_len] for prefix_len in metrics["n_cases"].keys()]) / n_cases

        # calculate the t metrics for the whole dataset
        metrics["acc_t"] = accuracy_score(merged_df["t_true"], merged_df["t_est"])
        metrics["f1_t"] = f1_score(merged_df["t_true"], merged_df["t_est"])
        metrics["precision_t"] = precision_score(merged_df["t_true"], merged_df["t_est"])
        metrics["recall_t"] = recall_score(merged_df["t_true"], merged_df["t_est"])
        metrics["roc_auc_t"] = roc_auc_score(merged_df["t_true"], merged_df["t_est"])
        metrics["pr_auc_t"] = average_precision_score(merged_df["t_true"], merged_df["t_est"])

        if ensemble:
            return metrics, merged_df
        return metrics