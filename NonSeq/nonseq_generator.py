from Seq import distributions
from NonSeq.base import BaseGenModel
import torch
from torch.utils import data
from itertools import chain
# from plotting.plotting import fig2img
from tqdm import tqdm
import matplotlib.pyplot as plt
from contextlib import contextmanager
from NonSeq.tarnet import TarNetFunctions
from NonSeq.adjusted_t_learner import TLearnerFunctions
from NonSeq.adjusted_s_learner import SLearnerFunctions


@contextmanager
def eval_ctx(mdl, debug=False, is_train=False):
    for net in mdl.networks: net.eval()
    torch.autograd.set_detect_anomaly(debug)
    with torch.set_grad_enabled(mode=is_train):
        yield
    torch.autograd.set_detect_anomaly(False)
    for net in mdl.networks: net.train()


# class MLPParams:
#     def __init__(self, n_hidden_layers=1, dim_h=64, activation=nn.ReLU()):
#         self.n_hidden_layers = n_hidden_layers
#         self.dim_h = dim_h
#         self.activation = activation


# _DEFAULT_MLP = dict(mlp_params_t_w=MLPParams(), mlp_params_y_tw=MLPParams())


# class TrainingParams:
#     def __init__(self, batch_size=32, lr=0.001, num_epochs=100, verbose=True, print_every_iters=100,
#                  eval_every=100, plot_every=100, p_every=100,
#                  optim=torch.optim.Adam, **optim_args):
#         self.batch_size = batch_size
#         self.lr = lr
#         self.num_epochs = num_epochs
#         self.verbose = verbose
#         self.print_every_iters = print_every_iters
#         self.optim = optim
#         self.eval_every = eval_every
#         self.plot_every = plot_every
#         self.p_every = p_every
#         self.optim_args = optim_args


class CausalDataset(data.Dataset):
    def __init__(self, x, t, y, xtype='float32', ttype='float32', ytype='float32'):
        self.x = x.to(torch.float32)
        self.t = t.to(torch.float32)
        self.y = y.to(torch.float32)
        self.y = self.y.unsqueeze(1) if len(self.y.size()) == 1 else self.y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return (
            self.x[index],
            self.t[index],
            self.y[index],
        )


# TODO: for more complex w, we might need to share parameters (dependent on the problem)
class NonSeqGenerator(BaseGenModel):
    def __init__(self, data_train, data_val, data_test, prep_utils, MODEL_PARAMS, inference=False):
        self.data_train = data_train
        self.data_val = data_val
        self.data_test = data_test
        self.prep_utils = prep_utils
        self.MODEL_PARAMS = MODEL_PARAMS
        self.inference = inference
        # Set seed
        torch.manual_seed(MODEL_PARAMS["seed"])
        torch.cuda.manual_seed(MODEL_PARAMS["seed"])
        torch.cuda.manual_seed_all(MODEL_PARAMS["seed"])

        # Set distribution
        if len(self.MODEL_PARAMS["y"]["outcome_distribution"]) > 1:
            if "atoms" in self.MODEL_PARAMS["y"]["outcome_distribution"]:
                self.outcome_distribution = distributions.MixedDistributionAtoms(atoms=self.prep_utils["atoms_scaled"], dist=distributions.SigmoidFlow(ndim=int(self.MODEL_PARAMS["dim_sigmoidflow"]), base_distribution=self.MODEL_PARAMS["y"]["outcome_distribution"][0]), bin_width=self.prep_utils["bin_width_scaled"], stdev_atoms = self.prep_utils["stdev_atoms_scaled"], atom_part="none")
        else:
            if self.MODEL_PARAMS["y"]["outcome_distribution"][0] == "bernoulli":
                self.outcome_distribution = distributions.Bernoulli()

        if len(self.MODEL_PARAMS["t"]["treatment_distribution"]) == 1:
            if self.MODEL_PARAMS["t"]["treatment_distribution"][0] == "bernoulli":
                self.treatment_distribution = distributions.Bernoulli()
            if self.MODEL_PARAMS["t"]["treatment_distribution"][0] == "categorical":
                    self.treatment_distribution = distributions.Categorical(num_classes=self.MODEL_PARAMS["t"]["dim_t"])

        # Set dimensions
        self.dim_x = self.data_train["X"].shape[1]
        # self.dim_x = len(self.prep_utils["column_names"]) - 1 - 1 # -1 for the outcome, -1 for the treatment
        self.dim_t = self.MODEL_PARAMS["t"]["dim_t"]
        self.dim_y = self.MODEL_PARAMS["y"]["dim_y"]
        self.dim_outcome_distribution = self.outcome_distribution.num_params
        self.dim_dense_t = self.MODEL_PARAMS["t"]["dim_hidden"]
        self.dim_dense_y = self.MODEL_PARAMS["y"]["dim_hidden"]
        # self.dims = {"dim_x": self.dim_x, "dim_t": self.dim_t, "dim_y": self.dim_y, "dim_outcome_distribution": self.dim_outcome_distribution, "dim_dense": self.dim_dense}
        self.dims = {"dim_x": self.dim_x, "dim_t": self.dim_t, "dim_y": self.dim_y, "dim_outcome_distribution": self.dim_outcome_distribution, "dim_dense_y": self.dim_dense_y, "dim_dense_t": self.dim_dense_t}

        # Set up causal functions and networks
        if MODEL_PARAMS["causal_type"] == "TarNet":
            self.causal_functions = TarNetFunctions(self.dims, MODEL_PARAMS, self.treatment_distribution, self.outcome_distribution)
        elif MODEL_PARAMS["causal_type"] == "T-Learner":
            self.causal_functions = TLearnerFunctions(self.dims, MODEL_PARAMS, self.treatment_distribution, self.outcome_distribution)
        elif MODEL_PARAMS["causal_type"] == "S-Learner":
            self.causal_functions = SLearnerFunctions(self.dims, MODEL_PARAMS, self.treatment_distribution, self.outcome_distribution)
        
        self.networks_y = self.causal_functions.networks_y
        self.network_t = self.causal_functions.model_t_x
        self.networks = self.causal_functions.networks

        if self.MODEL_PARAMS["loss_type"] == 'separate':
            print("Training separate models")
            self.optim_y = torch.optim.Adam(
                chain(*[net.parameters() for net in self.networks_y]),
                self.MODEL_PARAMS["y"]["lr"]
            )

            self.optim_t = torch.optim.Adam(
                self.network_t.parameters(),
                self.MODEL_PARAMS["t"]["lr"]
            )
            self.optim = None
        else:
            self.optim_t, self.optim_y = None, None
            self.optim = torch.optim.Adam(
                chain(*[net.parameters() for net in self.networks]),
                self.MODEL_PARAMS["t"]["lr"]
            )


    # def __init__(self, w, t, y, seed=1,
    #              network_params=None,
    #              training_params=TrainingParams(),
    #              binary_treatment=False,
    #              outcome_distribution: distributions.BaseDistribution = distributions.FactorialGaussian(),
    #              outcome_min=None,
    #              outcome_max=None,
    #              train_prop=1,
    #              val_prop=0,
    #              test_prop=0,
    #              shuffle=True,
    #              early_stop=True,
    #              patience=None,
    #              ignore_w=False,
    #              grad_norm=float('inf'),
    #              prep_utils=PlaceHolderTransform,
    #              w_transform=PlaceHolderTransform,
    #              t_transform=PlaceHolderTransform,
    #              y_transform=PlaceHolderTransform,
    #              savepath='.cache_best_model.pt',
    #              test_size=None,
    #              additional_args=dict()):
    #     super(MLP, self).__init__(*self._matricize((w, t, y)), seed=seed,
    #                               train_prop=train_prop, val_prop=val_prop,
    #                               test_prop=test_prop, shuffle=shuffle,
    #                               w_transform=w_transform,
    #                               t_transform=t_transform,
    #                               y_transform=y_transform,
    #                               test_size=test_size)

    #     self.binary_treatment = binary_treatment
    #     if binary_treatment:  # todo: input?
    #         self.treatment_distribution = distributions.Bernoulli()
    #     else:
    #         self.treatment_distribution = distributions.FactorialGaussian()
    #     self.outcome_distribution = outcome_distribution
    #     self.outcome_min = outcome_min
    #     self.outcome_max = outcome_max
    #     self.early_stop = early_stop
    #     self.patience = patience
    #     self.ignore_w = ignore_w
    #     self.grad_norm = grad_norm
    #     self.savepath = savepath
    #     self.additional_args = additional_args

    #     self.dim_w = self.w_transformed.shape[1]
    #     self.dim_t = self.t_transformed.shape[1]
    #     self.dim_y = self.y_transformed.shape[1]

    #     if network_params is None:
    #         network_params = _DEFAULT_MLP
    #     self.network_params = network_params
    #     self.build_networks()

    #     self.training_params = training_params
    #     self.optim = training_params.optim(
    #         chain(*[net.parameters() for net in self.networks]),
    #         training_params.lr,
    #         **training_params.optim_args
    #     )

    #     self.data_loader = data.DataLoader(
    #         CausalDataset(self.w_transformed, self.t_transformed, self.y_transformed),
    #         batch_size=training_params.batch_size,
    #         shuffle=True,
    #     )

    #     if len(self.val_idxs) > 0:
    #         self.data_loader_val = data.DataLoader(
    #             CausalDataset(
    #                 self.w_val_transformed,
    #                 self.t_val_transformed,
    #                 self.y_val_transformed,
    #             ),
    #             batch_size=training_params.batch_size,
    #             shuffle=True,
    #         )

    #     self.best_val_loss = float("inf")

    # def _matricize(self, data):
    #     return [np.reshape(d, [d.shape[0], -1]) for d in data]

    # def _build_mlp(self, dim_x, dim_y, MLP_params=MLPParams(), output_multiplier=2):
    #     dim_h = MLP_params.dim_h
    #     hidden_layers = [nn.Linear(dim_x, dim_h), MLP_params.activation]
    #     for _ in range(MLP_params.n_hidden_layers - 1):
    #         hidden_layers += [nn.Linear(dim_h, dim_h), MLP_params.activation]
    #     hidden_layers += [nn.Linear(dim_h, dim_y * output_multiplier)]
    #     return nn.Sequential(*hidden_layers)

    # def build_networks(self):
    #     self.MLP_params_t_w = self.network_params['mlp_params_t_w']
    #     self.MLP_params_y_tw = self.network_params['mlp_params_y_tw']
    #     output_multiplier_t = 1 if self.binary_treatment else 2
    #     self.mlp_t_w = self._build_mlp(self.dim_w, self.dim_t, self.MLP_params_t_w, output_multiplier_t)
    #     self.mlp_y_tw = self._build_mlp(self.dim_w + self.dim_t, self.dim_y, self.MLP_params_y_tw,
    #                                     self.outcome_distribution.num_params)
    #     self.networks = [self.mlp_t_w, self.mlp_y_tw]

    # def _get_loss(self, w, t, y):
    #     t_ = self.mlp_t_w(w)
    #     if self.ignore_w:
    #         w = torch.zeros_like(w)
    #     y_ = self.mlp_y_tw(torch.cat([w, t], dim=1))
    #     # check whether y_ contains nan values
    #     if torch.isnan(y_).any():
    #         print('y_ contains nan values')
    #     print('y_:', y_)
    #     loss_t = self.treatment_distribution.loss(t, t_)
    #     loss_y = self.outcome_distribution.loss(y, y_)
    #     loss = loss_t + loss_y
    #     return loss, loss_t, loss_y

    def train_joint(self):
        """
        Train the model, only used when loss_type is 'joint'
        """
        self.val_losses_y = []
        self.val_losses_t = []
        self.val_losses = []
        self.t_losses = []
        self.y_losses = []
        self.losses = []
        loss_val = float("inf")

        self.data_loader_train = data.DataLoader(
            CausalDataset(x=self.data_train["X"], t=self.data_train["T"], y=self.data_train["Y"]), shuffle=True, batch_size=self.MODEL_PARAMS["t"]["batch_size"])
        self.data_loader_val = data.DataLoader(
            CausalDataset(x=self.data_val["X"], t=self.data_val["T"], y=self.data_val["Y"]), shuffle=True, batch_size=self.MODEL_PARAMS["t"]["batch_size"])

        c = 0
        self.best_val_loss = float("inf")
        self.best_val_idx = 0
        for _ in tqdm(range(self.MODEL_PARAMS["num_epochs"]), disable=self.MODEL_PARAMS["disable_progress_bar"]):
            for x, t, y in self.data_loader_train:
                self.optim.zero_grad()
                loss, loss_t, loss_y = self.causal_functions.get_loss(x=x, t=t, y=y)
                self.losses.append(loss.item())
                self.t_losses.append(loss_t.item())
                self.y_losses.append(loss_y.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(chain(*[net.parameters() for net in self.networks]), self.MODEL_PARAMS["grad_norm"])
                self.optim.step()

                c += 1
                if c % self.MODEL_PARAMS["print_every_iters"] == 0:
                    print("\n")
                    print("Iteration :", c, "Epoch: ", _)
                    print('    Training loss:', loss.item())

                if c % self.MODEL_PARAMS["eval_every"] == 0:
                    with eval_ctx(self):
                        loss_val = self.evaluate(data_type="val")
                        self.val_losses.append(loss_val)
                    print("    Val loss:", loss_val)
                    if loss_val < self.best_val_loss:
                        self.best_val_loss = loss_val
                        self.best_val_idx = c
                        print("    saving best-val-loss-y model")
                        torch.save([net.state_dict() for net in self.networks], self.MODEL_PARAMS["savepath_joint"])

                # if c % self.MODEL_PARAMS["plot_every"] == 0:
                #     # use matplotlib
                #     plt.figure()
                #     plt.plot(self.losses, label='loss')
                #     plt.plot(self.val_losses, label='val_loss')
                #     plt.legend()
                #     plt.ioff()
                #     plt.close()

                #     with eval_ctx(self):
                #         plots = self.plot_ty_dists(verbose=False, dataset="train")
                #         plots_val = self.plot_ty_dists(verbose=False, dataset="val")
                #         plots_test = self.plot_ty_dists(verbose=False, dataset="test")
                        
                if c % self.MODEL_PARAMS["p_every"] == 0:
                    with eval_ctx(self):
                        uni_metrics_test = self.get_univariate_quant_metrics(dataset="test", verbose=False, outcome_distribution=self.outcome_distribution)
                        multi_variate_metrics_test = self.get_multivariate_quant_metrics(include_x=False, n_permutations=50, verbose=False, dataset="test", calculate_wssd=False)
                        print("    Test: ", uni_metrics_test)
                        print("    Test Multi: ", multi_variate_metrics_test)

            if self.MODEL_PARAMS["early_stop"] and self.MODEL_PARAMS["patience"] is not None and c - self.best_val_idx > self.MODEL_PARAMS["patience"]:
                print('early stopping criterion reached. Ending experiment.')
                # plt.figure()
                # plt.plot(self.t_losses, label='t_loss')
                # plt.plot(self.y_losses, label='y_loss')
                # plt.plot(self.val_losses, label='val_loss')
                # plt.legend()
                # plt.show()
                break

        if self.MODEL_PARAMS["early_stop"]:
            print("loading best-val-loss model (early stopping checkpoint)")
            for net, params in zip(self.networks, torch.load(self.MODEL_PARAMS["savepath_joint"])): net.load_state_dict(params)

    def train_t(self):
        """
        Train the model, only used when loss_type is 'separate'
        """
        self.val_losses_t = []
        self.t_losses = []
        self.losses_t = []
        loss_val_t = float("inf")

        self.data_loader_train = data.DataLoader(
            CausalDataset(x=self.data_train["X"], t=self.data_train["T"], y=self.data_train["Y"]), shuffle=True, batch_size=self.MODEL_PARAMS["t"]["batch_size"])
        self.data_loader_val = data.DataLoader(
            CausalDataset(x=self.data_val["X"], t=self.data_val["T"], y=self.data_val["Y"]), shuffle=True, batch_size=self.MODEL_PARAMS["t"]["batch_size"])

        c = 0
        self.best_val_loss_t = float("inf")
        self.best_val_idx_t = 0
        for _ in tqdm(range(self.MODEL_PARAMS["num_epochs"]), disable=self.MODEL_PARAMS["disable_progress_bar"]):
            for x, t, y in self.data_loader_train:
                self.optim_t.zero_grad()
                loss_t = self.causal_functions.get_loss_t_x(x=x, t=t)
                self.t_losses.append(loss_t.item())
                self.val_losses_t.append(loss_val_t)
                loss_t.backward()
                torch.nn.utils.clip_grad_norm_(self.network_t.parameters(), self.MODEL_PARAMS["grad_norm"])
                self.optim_t.step()

                c += 1
                if c % self.MODEL_PARAMS["print_every_iters"] == 0:
                    print("\n")
                    print("Iteration :", c, "Epoch: ", _)
                    print('    Training loss:', loss_t.item())

                if c % self.MODEL_PARAMS["eval_every"] == 0:
                    with eval_ctx(self):
                        loss_val_t = self.evaluate_t(data_type="val")
                    print("    Val loss t:", loss_val_t)
                    if loss_val_t < self.best_val_loss_t:
                        self.best_val_loss_t = loss_val_t
                        self.best_val_idx_t = c
                        print("    saving best-val-loss-t model")
                        torch.save(self.network_t.state_dict(), self.MODEL_PARAMS["savepath_t"])

                # if c % self.MODEL_PARAMS["plot_every"] == 0:
                #     # use matplotlib
                #     plt.figure()
                #     plt.plot(self.losses_t, label='loss_t')
                #     plt.plot(self.val_losses_t, label='val_loss_t')
                #     plt.legend()
                #     plt.ioff()
                #     plt.close()

                #     with eval_ctx(self):
                #         plots = self.plot_ty_dists(verbose=False, dataset="train")
                #         plots_val = self.plot_ty_dists(verbose=False, dataset="val")
                #         plots_test = self.plot_ty_dists(verbose=False, dataset="test")
                        
                if c % self.MODEL_PARAMS["p_every"] == 0:
                    with eval_ctx(self):
                        uni_metrics_test = self.get_univariate_quant_metrics(dataset="test", verbose=False, outcome_distribution=self.outcome_distribution)
                        multi_variate_metrics_test = self.get_multivariate_quant_metrics(include_w=False, n_permutations=50, verbose=False, dataset="test", calculate_wssd=False)
                        print("    Test: ", uni_metrics_test)
                        print("    Test Multi: ", multi_variate_metrics_test)

            if self.MODEL_PARAMS["early_stop"] and self.MODEL_PARAMS["patience"] is not None and c - self.best_val_idx_t > self.MODEL_PARAMS["patience"]:
                print('early stopping criterion reached. Ending experiment.')
                # plt.figure()
                # plt.plot(self.losses_t, label='loss_t')
                # plt.plot(self.val_losses_t, label='val_loss_t')
                # plt.legend()
                # plt.show()
                break
        
        if self.MODEL_PARAMS["early_stop"]:
            print("loading best-val-loss model (early stopping checkpoint)")
            self.network_t.load_state_dict(torch.load(self.MODEL_PARAMS["savepath_t"]))

    def train_y(self):
        """
        Train the model, only used when loss_type is 'separate'
        """
        self.val_losses_y = []
        self.y_losses = []
        self.losses_y = []
        loss_val_y = float("inf")

        self.data_loader_train = data.DataLoader(
            CausalDataset(x=self.data_train["X"], t=self.data_train["T"], y=self.data_train["Y"]), shuffle=True, batch_size=self.MODEL_PARAMS["y"]["batch_size"])
        self.data_loader_val = data.DataLoader(
            CausalDataset(x=self.data_val["X"], t=self.data_val["T"], y=self.data_val["Y"]), shuffle=True, batch_size=self.MODEL_PARAMS["y"]["batch_size"])

        c = 0
        self.best_val_loss_y = float("inf")
        self.best_val_idx_y = 0
        for _ in tqdm(range(self.MODEL_PARAMS["num_epochs"]), disable=self.MODEL_PARAMS["disable_progress_bar"]):
            for x, t, y in self.data_loader_train:
                self.optim_y.zero_grad()
                loss_y = self.causal_functions.get_loss_y_t_x(x=x, t=t, y=y)
                self.y_losses.append(loss_y.item())
                self.val_losses_y.append(loss_val_y)
                loss_y.backward()
                torch.nn.utils.clip_grad_norm_(chain(*[net.parameters() for net in self.networks_y]), self.MODEL_PARAMS["grad_norm"])
                self.optim_y.step()

                c += 1
                if c % self.MODEL_PARAMS["print_every_iters"] == 0:
                    print("\n")
                    print("Iteration :", c, "Epoch: ", _)
                    print('    Training loss:', loss_y.item())

                if c % self.MODEL_PARAMS["eval_every"] == 0:
                    with eval_ctx(self):
                        loss_val_y = self.evaluate_y(data_type="val")
                    print("    Val loss y:", loss_val_y)
                    if loss_val_y < self.best_val_loss_y:
                        self.best_val_loss_y = loss_val_y
                        self.best_val_idx_y = c
                        print("    saving best-val-loss-y model")
                        torch.save([net.state_dict() for net in self.networks_y], self.MODEL_PARAMS["savepath_y"])

                # if c % self.MODEL_PARAMS["plot_every"] == 0:
                #     # use matplotlib
                #     plt.figure()
                #     plt.plot(self.losses_y, label='loss_y')
                #     plt.plot(self.val_losses_y, label='val_loss_y')
                #     plt.legend()
                #     plt.ioff()
                #     plt.close()

                #     with eval_ctx(self):
                #         plots = self.plot_ty_dists(verbose=False, dataset="train")
                #         plots_val = self.plot_ty_dists(verbose=False, dataset="val")
                #         plots_test = self.plot_ty_dists(verbose=False, dataset="test")
                        
                if c % self.MODEL_PARAMS["p_every"] == 0:
                    with eval_ctx(self):
                        uni_metrics_test = self.get_univariate_quant_metrics(dataset="test", verbose=False, outcome_distribution=self.outcome_distribution)
                        multi_variate_metrics_test = self.get_multivariate_quant_metrics(include_w=False, n_permutations=50, verbose=False, dataset="test", calculate_wssd=False)
                        print("    Test: ", uni_metrics_test)
                        print("    Test Multi: ", multi_variate_metrics_test)

            if self.MODEL_PARAMS["early_stop"] and self.MODEL_PARAMS["patience"] is not None and c - self.best_val_idx_y > self.MODEL_PARAMS["patience"]:
                print('early stopping criterion reached. Ending experiment.')
                # plt.figure()
                # plt.plot(self.losses_y, label='loss_y')
                # plt.plot(self.val_losses_y, label='val_loss_y')
                # plt.legend()
                # plt.show()
                break
        
        if self.MODEL_PARAMS["early_stop"]:
            print("loading best-val-loss model (early stopping checkpoint)")
            for net, params in zip(self.networks_y, torch.load(self.MODEL_PARAMS["savepath_y"])): net.load_state_dict(params)

    # def train(self, early_stop=None, print_=lambda s, print_: print(s), comet_exp=None):
    #     self.losses = []
    #     self.val_losses = []
    #     self.t_losses = []
    #     self.y_losses = []
    #     loss_val = float("inf")
    #     if early_stop is None:
    #         early_stop = self.early_stop

    #     c = 0
    #     self.best_val_loss = float("inf")
    #     self.best_val_idx = 0
    #     for _ in tqdm(range(self.MODEL_PARAMS["num_epochs"])):
    #         for w, t, y in self.data_loader:
    #             self.optim.zero_grad()
    #             loss, loss_t, loss_y = self._get_loss(w, t, y)
    #             self.losses.append(loss.item())
    #             self.t_losses.append(loss_t.item())
    #             self.y_losses.append(loss_y.item())
    #             self.val_losses.append(loss_val)
    #             # TODO: learning rate can be separately adjusted by weighting the losses here
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(chain(*[net.parameters() for net in self.networks]), self.grad_norm)
    #             self.optim.step()

    #             c += 1
    #             if self.MODEL_PARAMS.verbose and c % self.MODEL_PARAMS.print_every_iters == 0:
    #                 print("\n")
    #                 print("Iteration :", c)
    #                 print('    Training loss:', loss.item())

    #                 if comet_exp is not None:
    #                     comet_exp.log_metric("loss_t", loss_t.item())
    #                     comet_exp.log_metric("loss_y", loss_y.item())

    #             if c % self.MODEL_PARAMS.eval_every == 0 and len(self.val_idxs) > 0:
    #                 with eval_ctx(self):
    #                     loss_val = self.evaluate(self.data_loader_val, only_y_loss=True).item()
    #                 if comet_exp is not None:
    #                     comet_exp.log_metric('loss_val', loss_val)
    #                 print("    Val loss:", loss_val)
    #                 if loss_val < self.best_val_loss:
    #                     self.best_val_loss = loss_val
    #                     self.best_val_idx = c
    #                     print("    saving best-val-loss model")
    #                     torch.save([net.state_dict() for net in self.networks], self.savepath)

    #             if c % self.MODEL_PARAMS.plot_every == 0:
    #                 # use matplotlib
    #                 plt.figure()
    #                 plt.plot(self.losses, label='loss')
    #                 plt.plot(self.t_losses, label='t_loss')
    #                 plt.plot(self.y_losses, label='y_loss')
    #                 plt.plot(self.val_losses, label='val_loss')
    #                 plt.legend()
    #                 plt.ioff()
    #                 plt.close()


    #                 with eval_ctx(self):
    #                     plots = self.plot_ty_dists(verbose=False, dataset="train")
    #                     plots_val = self.plot_ty_dists(verbose=False, dataset="val")
    #                     plots_test = self.plot_ty_dists(verbose=False, dataset="test")
                        
    #             if c % self.MODEL_PARAMS.p_every == 0:
    #                 with eval_ctx(self):
    #                     uni_metrics_test = self.get_univariate_quant_metrics(dataset="test", verbose=False, outcome_distribution=self.outcome_distribution)
    #                     multi_variate_metrics_test = self.get_multivariate_quant_metrics(include_w=False, n_permutations=50, verbose=False, dataset="test", calculate_wssd=False)
    #                     print("    Test: ", uni_metrics_test)
    #                     print("    Test Multi: ", multi_variate_metrics_test)

                
    #         if early_stop and self.patience is not None and c - self.best_val_idx > self.patience:
    #             print('early stopping criterion reached. Ending experiment.')
    #             plt.figure()
    #             plt.plot(self.losses, label='loss')
    #             plt.plot(self.val_losses, label='val_loss')
    #             plt.legend()
    #             plt.show()
    #             break

    #     if early_stop and len(self.val_idxs) > 0:
    #         print("loading best-val-loss model (early stopping checkpoint)")
    #         for net, params in zip(self.networks, torch.load(self.savepath)):
    #             net.load_state_dict(params)

    def evaluate(self, data_type):
        if data_type == "train":
            data_loader = self.data_loader_train
        elif data_type == "val":
            data_loader = self.data_loader_val

        n = 0
        loss = 0
        for x, t, y in data_loader:
            loss += self.causal_functions.get_loss(x=x, t=t, y=y)[0] * x.size(0)
            n += x.size(0)
            print(x.size(0))
        return ( loss / n ).item()
    
    def evaluate_t(self, data_type):
        if data_type == "train":
            data_loader = self.data_loader_train
        elif data_type == "val":
            data_loader = self.data_loader_val

        n = 0
        loss = 0
        for x, t, y in data_loader:
            loss += self.causal_functions.get_loss_t_x(x=x, t=t) * x.size(0)
            n += x.size(0)
        return ( loss / n ).item()
    
    def evaluate_y(self, data_type):
        if data_type == "train":
            data_loader = self.data_loader_train
        elif data_type == "val":
            data_loader = self.data_loader_val

        n = 0
        loss = 0
        for x, t, y in data_loader:
            loss += self.causal_functions.get_loss_y_t_x(x=x, t=t, y=y) * x.size(0)
            n += x.size(0)
        return ( loss / n ).item()

    # def evaluate(self, data_loader, only_y_loss=False):
    #     loss = 0
    #     n = 0
    #     for w, t, y in data_loader:
    #         if only_y_loss:
    #             loss += self._get_loss(w, t, y)[2] * w.size(0)
    #         else:
    #             loss += self._get_loss(w, t, y)[0] * w.size(0)
    #         n += w.size(0)
    #     return loss / n

    def _sample_t(self, x, overlap=1):
        t_ = self.causal_functions.forward_t_x(x=x)
        t_indices = self.treatment_distribution.sample(t_, overlap=overlap)
        # make sure that the treatment is binary (so if t = 2 for example, we get [0, 0, 1])
        if self.MODEL_PARAMS["t"]["dim_t"] > 1:
            t_samples = torch.eye(self.MODEL_PARAMS["t"]["dim_t"])[t_indices.long()].squeeze()
        else:
            t_samples = t_indices
        return t_samples

    # def _sample_t(self, w=None, overlap=1):
    #     t_ = self.mlp_t_w(torch.from_numpy(w).float())
    #     return self.treatment_distribution.sample(t_, overlap=overlap)

    def _sample_y(self, t, x=None, ret_counterfactuals=False):
        if self.MODEL_PARAMS["ignore_x"]:
            x = torch.zeros_like(x)
        
        if ret_counterfactuals:
            y_total_ = self.causal_functions.forward_y_t_x(x=x, t=t, ret_counterfactuals=True)
            y_samples_total = [torch.tensor(self.outcome_distribution.sample(y_)) for y_ in y_total_]
            return y_samples_total
        else:
            y_ = self.causal_functions.forward_y_t_x(x=x, t=t, ret_counterfactuals=False)
            y_samples = self.outcome_distribution.sample(y_)
            return y_samples

    def mean_y(self, t, x):
        if self.MODEL_PARAMS["ignore_x"]:
            x = torch.zeros_like(x)
        return self.outcome_distribution.mean(self.causal_functions.forward_y_t_x(x=x, t=t))
    
    def val(self, t_only=False, y_only=False):
        # Set in evaluation mode

        if self.MODEL_PARAMS["loss_type"] == 'joint':
            for net in self.networks: net.eval()
        else:
            for net in self.networks_y: net.eval()
            self.network_t.eval()

        uni_metrics_val = self.get_univariate_quant_metrics(dataset="val", verbose=False, outcome_distribution=self.outcome_distribution,t_only=t_only, y_only=y_only)

        print('\n')
        print("Univariate metrics val: ", uni_metrics_val)

        if self.MODEL_PARAMS["loss_type"] == 'joint':
            for net in self.networks: net.train()
        else:
            for net in self.networks_y: net.train()
            self.network_t.train()
        return uni_metrics_val
    
    # def evaluate_statistical(self, only_univariate=False):
    #     # Set in evaluation mode
    #     if self.MODEL_PARAMS["loss_type"] == 'joint':
    #         for net in self.networks: net.eval()
    #     else:
    #         for net in self.networks_y: net.eval()
    #         self.network_t.eval()


    #     uni_metrics_test = self.get_univariate_quant_metrics(dataset="test", verbose=False, outcome_distribution=self.outcome_distribution)
    #     multi_metrics_test = {}
    #     if not only_univariate:
    #         multi_metrics_test = self.get_multivariate_quant_metrics(dataset="test", verbose=False, include_x=False, n_permutations=1000)

    #     print('\n')
    #     print("Univariate metrics test: ", uni_metrics_test)
    #     print("Multivariate metrics test: ", multi_metrics_test)

    #     return uni_metrics_test, multi_metrics_test
    
    def evaluate_statistical(self, only_univariate=False, t_model=None, y_model=None):
        # Set in evaluation mode
        for net in self.networks_y: net.eval()
        self.network_t.eval()

        uni_metrics_test = self.get_univariate_quant_metrics(dataset="test", verbose=False, outcome_distribution=self.outcome_distribution, t_model=t_model, y_model=y_model)
        multi_metrics_test_no_x = {}
        multi_metrics_test = {}
        if not only_univariate:
            multi_metrics_test_no_x = self.get_multivariate_quant_metrics(dataset="test", verbose=False, include_x=False, n_permutations=1000, n=500, t_model=t_model, y_model=y_model)
            multi_metrics_test = self.get_multivariate_quant_metrics(dataset="test", verbose=False, include_x=True, n_permutations=1000, n=500, t_model=t_model, y_model=y_model)

        print('\n')
        print("Univariate metrics test: ", uni_metrics_test)
        print("Multivariate metrics test: ", multi_metrics_test)

        return uni_metrics_test, multi_metrics_test_no_x, multi_metrics_test

    # def _sample_y(self, t, w=None, ret_counterfactuals=False):
    #     if self.ignore_w:
    #         w = np.zeros_like(w)
    #     wt = np.concatenate([w, t], 1)
    #     if ret_counterfactuals:
    #         y0_, y1_ = self.mlp_y_tw(torch.from_numpy(wt).float(), ret_counterfactuals=True)
    #         y0_samples = self.outcome_distribution.sample(y0_)
    #         y1_samples = self.outcome_distribution.sample(y1_)
    #         if self.outcome_min is not None or self.outcome_max is not None:
    #             y0_samples = np.clip(y0_samples, self.outcome_min, self.outcome_max)
    #             y1_samples = np.clip(y1_samples, self.outcome_min, self.outcome_max)
    #         return y0_samples, y1_samples
    #     else:
    #         y_ = self.mlp_y_tw(torch.from_numpy(wt).float(), ret_counterfactuals=False)
    #         y_samples = self.outcome_distribution.sample(y_)
    #         if self.outcome_min is not None or self.outcome_max is not None:
    #             y_samples = np.clip(y_samples, self.outcome_min, self.outcome_max)
    #         return y_samples

    # def mean_y(self, t, w):
    #     if self.ignore_w:
    #         w = np.zeros_like(w)
    #     wt = np.concatenate([w, t], 1)
    #     return self.outcome_distribution.mean(self.mlp_y_tw(torch.from_numpy(wt).float()))