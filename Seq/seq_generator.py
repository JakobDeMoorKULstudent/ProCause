from Seq.t_learner import TLearnerFunctions
from Seq.s_learner import SLearnerFunctions
from Seq.tarnet import TarNetFunctions
import numpy as np
from Seq.distributions import distributions
from Seq.base import BaseGenModel
import torch
from torch.utils import data
from itertools import chain
from tqdm import tqdm
import matplotlib.pyplot as plt
from contextlib import contextmanager

@contextmanager
def eval_ctx_y(mdl, debug=False, is_train=False):
    for net in mdl.networks_y: net.eval()
    torch.autograd.set_detect_anomaly(debug)
    with torch.set_grad_enabled(mode=is_train):
        yield
    torch.autograd.set_detect_anomaly(False)
    for net in mdl.networks_y: net.train()

@contextmanager
def eval_ctx_t(mdl, debug=False, is_train=False):
    mdl.network_t.eval()
    torch.autograd.set_detect_anomaly(debug)
    with torch.set_grad_enabled(mode=is_train):
        yield
    torch.autograd.set_detect_anomaly(False)
    mdl.network_t.train()

class CausalDataset(data.Dataset):
    def __init__(self, x_case, x_event, prefix_len, t, y, xtype='float32', ttype='float32', ytype='float32'):
        self.x_case = x_case.to(torch.float32)
        self.x_event = x_event.to(torch.float32)
        self.prefix_len = prefix_len
        self.t = t.to(torch.float32)
        # Make sure the "Y"'s have the right shape ([n, 1] instead of [n])
        self.y = y.to(torch.float32)
        self.y = self.y.unsqueeze(1) if len(self.y.size()) == 1 else self.y

    def __len__(self):
        return self.x_event.size(0)

    def __getitem__(self, index):
        return (
            self.x_case[index],
            self.x_event[index],
            self.prefix_len[index],
            self.t[index],
            self.y[index],
        )

class SeqGenerator(BaseGenModel):
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
        self.dim_x_case = len(self.prep_utils["case_cols_encoded"])
        self.dim_x_event = len(self.prep_utils["event_cols_encoded"])
        self.dim_t = self.MODEL_PARAMS["t"]["dim_t"]
        self.dim_y = self.MODEL_PARAMS["y"]["dim_y"]
        self.dim_outcome_distribution = self.outcome_distribution.num_params
        self.dim_dense_t = self.MODEL_PARAMS["t"]["dim_hidden"]
        self.dim_lstm_t = self.MODEL_PARAMS["t"]["dim_hidden"]
        self.dim_dense_y = self.MODEL_PARAMS["y"]["dim_hidden"]
        self.dim_lstm_y = self.MODEL_PARAMS["y"]["dim_hidden"]
        self.dims = {"dim_x_case": self.dim_x_case, "dim_x_event": self.dim_x_event, "dim_t": self.dim_t, "dim_y": self.dim_y, "dim_outcome_distribution": self.dim_outcome_distribution, "dim_dense_t": self.dim_dense_t, "dim_lstm_t": self.dim_lstm_t, "dim_dense_y": self.dim_dense_y, "dim_lstm_y": self.dim_lstm_y}

        # Set up causal functions and networks
        if MODEL_PARAMS["causal_type"] == "T-Learner":
            self.causal_functions = TLearnerFunctions(self.dims, MODEL_PARAMS, self.treatment_distribution, self.outcome_distribution)
        elif MODEL_PARAMS["causal_type"] == "S-Learner":
            self.causal_functions = SLearnerFunctions(self.dims, MODEL_PARAMS, self.treatment_distribution, self.outcome_distribution)
        elif MODEL_PARAMS["causal_type"] == "TarNet":
            self.causal_functions = TarNetFunctions(self.dims, MODEL_PARAMS, self.treatment_distribution, self.outcome_distribution)
        # self.causal_functions = TarNetFunctions(self.dims, MODEL_PARAMS, self.treatment_distribution, self.outcome_distribution)
        
        self.networks_y = self.causal_functions.networks_y
        self.network_t = self.causal_functions.model_t_x

        self.optim_y = torch.optim.Adam(
            chain(*[net.parameters() for net in self.networks_y]),
            self.MODEL_PARAMS["y"]["lr"]
        )

        self.optim_t = torch.optim.Adam(
            self.network_t.parameters(),
            self.MODEL_PARAMS["t"]["lr"]
        )

        # if not self.inference:
        #     self.data_loader_train = data.DataLoader(
        #         CausalDataset(x_case=self.data_train["X_case"], x_event=self.data_train["X_event"], prefix_len=self.data_train["prefix_len"], t=self.data_train["T"], y=self.data_train["Y"]),
        #         batch_size=self.MODEL_PARAMS["batch_size"],
        #         shuffle=True,
        #     )

        #     self.data_loader_val = data.DataLoader(
        #         CausalDataset(x_case=self.data_val["X_case"], x_event=self.data_val["X_event"], prefix_len=self.data_val["prefix_len"], t=self.data_val["T"], y=self.data_val["Y"]),
        #         batch_size=self.MODEL_PARAMS["batch_size"],
        #         shuffle=True,
        #     )


    def _matricize(self, data):
        return [np.reshape(d, [d.shape[0], -1]) for d in data]
    
    def train_t(self):
        self.val_losses_t = []
        self.t_losses = []
        loss_val_t = float("inf")

        self.data_loader_train = data.DataLoader(
            CausalDataset(x_case=self.data_train["X_case"], x_event=self.data_train["X_event"], prefix_len=self.data_train["prefix_len"], t=self.data_train["T"], y=self.data_train["Y"]),
            batch_size=self.MODEL_PARAMS["t"]["batch_size"],
            shuffle=True,
        )
        self.data_loader_val = data.DataLoader(
            CausalDataset(x_case=self.data_val["X_case"], x_event=self.data_val["X_event"], prefix_len=self.data_val["prefix_len"], t=self.data_val["T"], y=self.data_val["Y"]),
            batch_size=self.MODEL_PARAMS["t"]["batch_size"],
            shuffle=True,
        )

        c = 0
        self.best_val_loss_t = float("inf")
        self.best_val_idx_t = 0
        for _ in tqdm(range(self.MODEL_PARAMS["num_epochs"]), disable=self.MODEL_PARAMS["disable_progress_bar"]):
            for x_case, x_event, prefix_len, t, y in self.data_loader_train:
                self.optim_t.zero_grad()
                loss_t = self.causal_functions.get_loss_t_x(x_case=x_case, x_event=x_event, t=t, prefix_len=prefix_len)
                self.t_losses.append(loss_t.item())
                loss_t.backward()
                torch.nn.utils.clip_grad_norm_(self.network_t.parameters(), self.MODEL_PARAMS["grad_norm"])
                self.optim_t.step()

                c += 1
                if c % self.MODEL_PARAMS["print_every_iters"] == 0:
                    print("\n")
                    print("Iteration :", c, "Epoch: ", _)
                    print('    Training loss t:', loss_t.item())

                if c % self.MODEL_PARAMS["eval_every"] == 0:
                    with eval_ctx_t(self):
                        loss_val_t = self.evaluate_t(data_type="val")
                    self.val_losses_t.append(loss_val_t)
                    print("    Val loss t:", loss_val_t) 
                    if loss_val_t < self.best_val_loss_t:
                        self.best_val_loss_t = loss_val_t
                        self.best_val_idx_t = c
                        print("    saving best-val-loss-t model")
                        torch.save(self.network_t.state_dict(), self.MODEL_PARAMS["savepath_t"])

                # if c % self.MODEL_PARAMS["plot_every"] == 0:
                #     # use matplotlib
                #     plt.figure()
                #     plt.plot(self.t_losses, label='t_loss')
                #     plt.plot(self.val_losses_t, label='val_t_loss')
                #     plt.legend()
                #     plt.ioff()
                #     plt.close()

                if c % self.MODEL_PARAMS["p_every"] == 0:
                    with eval_ctx_t(self):
                        uni_metrics_test = self.get_univariate_quant_metrics(dataset="test", verbose=False, outcome_distribution=self.outcome_distribution)
                        multi_variate_metrics_test = self.get_multivariate_quant_metrics(include_x=False, n_permutations=50, verbose=False, dataset="test", calculate_wssd=False)
                        print("    Test: ", uni_metrics_test)
                        print("    Test Multi: ", multi_variate_metrics_test)
                    
            if self.MODEL_PARAMS["early_stop"] and self.MODEL_PARAMS["patience"] is not None and c - self.best_val_idx_t > self.MODEL_PARAMS["patience"]:
                print('early stopping criterion reached. Ending experiment. ')
                # plt.figure()
                # plt.plot(self.t_losses, label='t_loss')
                # plt.plot(self.val_losses_t, label='val_t_loss')
                # plt.show()
                break
        
        if self.MODEL_PARAMS["early_stop"]:
            print("loading best-val-loss model (early stopping checkpoint)")
            self.network_t.load_state_dict(torch.load(self.MODEL_PARAMS["savepath_t"]))
    
    def train_y(self):
        self.val_losses_y = []
        self.y_losses = []
        loss_val_y = float("inf")

        self.data_loader_train = data.DataLoader(
            CausalDataset(x_case=self.data_train["X_case"], x_event=self.data_train["X_event"], prefix_len=self.data_train["prefix_len"], t=self.data_train["T"], y=self.data_train["Y"]),
            batch_size=self.MODEL_PARAMS["y"]["batch_size"],
            shuffle=True,
        )
        self.data_loader_val = data.DataLoader(
            CausalDataset(x_case=self.data_val["X_case"], x_event=self.data_val["X_event"], prefix_len=self.data_val["prefix_len"], t=self.data_val["T"], y=self.data_val["Y"]),
            batch_size=self.MODEL_PARAMS["y"]["batch_size"],
            shuffle=True,
        )

        c = 0
        self.best_val_loss_y = float("inf")
        self.best_val_idx_y = 0
        for _ in tqdm(range(self.MODEL_PARAMS["num_epochs"]), disable=self.MODEL_PARAMS["disable_progress_bar"]):
            for x_case, x_event, prefix_len, t, y in self.data_loader_train:
                self.optim_y.zero_grad()
                loss_y = self.causal_functions.get_loss_y_t_x(x_case=x_case, x_event=x_event, t=t, prefix_len=prefix_len, y=y)
                self.y_losses.append(loss_y.item())
                loss_y.backward()
                torch.nn.utils.clip_grad_norm_(chain(*[net.parameters() for net in self.networks_y]), self.MODEL_PARAMS["grad_norm"])
                self.optim_y.step()

                c += 1
                if c % self.MODEL_PARAMS["print_every_iters"] == 0:
                    print("\n")
                    print("Iteration :", c, "Epoch: ", _)
                    print('    Training loss y:', loss_y.item())

                if c % self.MODEL_PARAMS["eval_every"] == 0:
                    with eval_ctx_y(self):
                        loss_val_y = self.evaluate_y(data_type="val")
                    self.val_losses_y.append(loss_val_y)
                    print("    Val loss y:", loss_val_y)
                    if loss_val_y < self.best_val_loss_y:
                        self.best_val_loss_y = loss_val_y
                        self.best_val_idx_y = c
                        print("    saving best-val-loss-y model")
                        torch.save([net.state_dict() for net in self.networks_y], self.MODEL_PARAMS["savepath_y"])

                # if c % self.MODEL_PARAMS["plot_every"] == 0:
                #     # use matplotlib
                #     plt.figure()
                #     plt.plot(self.y_losses, label='y_loss')
                #     plt.plot(self.val_losses_y, label='val_y_loss')
                #     plt.legend()
                #     plt.ioff()
                #     plt.close()

                if c % self.MODEL_PARAMS["p_every"] == 0:
                    with eval_ctx_y(self):
                        uni_metrics_test = self.get_univariate_quant_metrics(dataset="test", verbose=False, outcome_distribution=self.outcome_distribution)
                        multi_variate_metrics_test = self.get_multivariate_quant_metrics(include_x=False, n_permutations=50, verbose=False, dataset="test", calculate_wssd=False)
                        print("    Test: ", uni_metrics_test)
                        print("    Test Multi: ", multi_variate_metrics_test)
            
            if self.MODEL_PARAMS["early_stop"] and self.MODEL_PARAMS["patience"] is not None and c - self.best_val_idx_y > self.MODEL_PARAMS["patience"]:
                print('early stopping criterion reached. Ending experiment. ')
                # plt.figure()
                # plt.plot(self.y_losses, label='y_loss')
                # plt.plot(self.val_losses_y, label='val_y_loss')
                # plt.show()
                break
        
        if self.MODEL_PARAMS["early_stop"]:
            print("loading best-val-loss model (early stopping checkpoint)")
            for net, params in zip(self.networks_y, torch.load(self.MODEL_PARAMS["savepath_y"])):
                net.load_state_dict(params)

    # def train(self):
    #     self.val_losses_y = []
    #     self.val_losses_t = []
    #     self.t_losses = []
    #     self.y_losses = []
    #     loss_val_y = float("inf")
    #     loss_val_t = float("inf")

    #     c = 0
    #     self.best_val_loss_y = float("inf")
    #     self.best_val_loss_t = float("inf")
    #     self.best_val_idx_y = 0
    #     self.best_val_idx_t = 0
    #     for _ in tqdm(range(self.MODEL_PARAMS["num_epochs"])):
    #         for x_case, x_event, prefix_len, t, y in self.data_loader_train:
    #             self.optim_y.zero_grad()
    #             loss_y = self.causal_functions.get_loss_y_t_x(x_case=x_case, x_event=x_event, t=t, prefix_len=prefix_len, y=y)
    #             self.y_losses.append(loss_y.item())
    #             loss_y.backward()
    #             torch.nn.utils.clip_grad_norm_(chain(*[net.parameters() for net in self.networks_y]), self.MODEL_PARAMS["grad_norm"])
    #             self.optim_y.step()

    #             if not self.MODEL_PARAMS["t_already_trained"]:
    #                 self.optim_t.zero_grad()
    #                 loss_t = self.causal_functions.get_loss_t_x(x_case=x_case, x_event=x_event, t=t, prefix_len=prefix_len)
    #                 self.t_losses.append(loss_t.item())
    #                 loss_t.backward()
    #                 torch.nn.utils.clip_grad_norm_(self.network_t.parameters(), self.MODEL_PARAMS["grad_norm"])
    #                 self.optim_t.step()

    #             c += 1
    #             if c % self.MODEL_PARAMS["print_every_iters"] == 0:
    #                 print("\n")
    #                 print("Iteration :", c)
    #                 print('    Training loss y:', loss_y.item())
    #                 if not self.MODEL_PARAMS["t_already_trained"]:
    #                     print('    Training loss t:', loss_t.item())

    #             if c % self.MODEL_PARAMS["eval_every"] == 0:
    #                 with eval_ctx(self):
    #                     loss_val_y, loss_val_t = self.evaluate(data_type="val")
    #                 self.val_losses_y.append(loss_val_y)
    #                 self.val_losses_t.append(loss_val_t)
    #                 print("    Val loss y:", loss_val_y)
    #                 if loss_val_y < self.best_val_loss_y:
    #                     self.best_val_loss_y = loss_val_y
    #                     self.best_val_idx_y = c
    #                     print("    saving best-val-loss-y model")
    #                     torch.save([net.state_dict() for net in self.networks_y], self.MODEL_PARAMS["savepath_y"])
    #                 if not self.MODEL_PARAMS["t_already_trained"]:
    #                     print("    Val loss t:", loss_val_t) 
    #                     if loss_val_t < self.best_val_loss_t:
    #                         self.best_val_loss_t = loss_val_t
    #                         self.best_val_idx_t = c
    #                         print("    saving best-val-loss-t model")
    #                         torch.save(self.network_t.state_dict(), self.MODEL_PARAMS["savepath_t"])

    #             if c % self.MODEL_PARAMS["plot_every"] == 0:
    #                 # use matplotlib
    #                 plt.figure()
    #                 plt.plot(self.t_losses, label='t_loss')
    #                 plt.plot(self.y_losses, label='y_loss')
    #                 plt.plot(self.val_losses_y, label='val_y_loss')
    #                 plt.plot(self.val_losses_t, label='val_t_loss')
    #                 plt.legend()
    #                 plt.ioff()
    #                 plt.close()

    #                 with eval_ctx(self):
    #                     plots = self.plot_ty_dists(verbose=False, dataset="train")
    #                     plots_val = self.plot_ty_dists(verbose=False, dataset="val")
    #                     plots_test = self.plot_ty_dists(verbose=False, dataset="test")
                        
    #             if c % self.MODEL_PARAMS["p_every"] == 0:
    #                 with eval_ctx(self):
    #                     uni_metrics_test = self.get_univariate_quant_metrics(dataset="test", verbose=False, outcome_distribution=self.outcome_distribution)
    #                     multi_variate_metrics_test = self.get_multivariate_quant_metrics(include_x=False, n_permutations=50, verbose=False, dataset="test", calculate_wssd=False)
    #                     print("    Test: ", uni_metrics_test)
    #                     print("    Test Multi: ", multi_variate_metrics_test)

    #         if self.MODEL_PARAMS["early_stop"] and self.MODEL_PARAMS["patience"] is not None and c - self.best_val_idx > self.MODEL_PARAMS["patience"]:
    #             print('early stopping criterion reached. Ending experiment.')
    #             plt.figure()
    #             plt.plot(self.t_losses, label='t_loss')
    #             plt.plot(self.y_losses, label='y_loss')
    #             plt.plot(self.val_losses_y, label='val_y_loss')
    #             plt.plot(self.val_losses_t, label='val_t_loss')
    #             plt.legend()
    #             plt.show()
    #             break

    #     if self.MODEL_PARAMS["early_stop"]:
    #         print("loading best-val-loss model (early stopping checkpoint)")
    #         for net, params in zip(self.networks_y, torch.load(self.MODEL_PARAMS["savepath_y"])):
    #             net.load_state_dict(params)
    #         if not self.MODEL_PARAMS["t_already_trained"]:
    #             self.network_t.load_state_dict(torch.load(self.MODEL_PARAMS["savepath_t"]))

    def evaluate_t(self, data_type):
        if data_type == "train":
            data_loader = self.data_loader_train
        elif data_type == "val":
            data_loader = self.data_loader_val

        loss_t = 0
        n = 0
        for x_case, x_event, prefix_len, t, y in data_loader:
            loss_t += self.causal_functions.get_loss_t_x(x_case=x_case, x_event=x_event, t=t, prefix_len=prefix_len) * x_event.size(0)
            n += x_event.size(0)
        return (loss_t / n).item()

    def evaluate_y(self, data_type):
        if data_type == "train":
            data_loader = self.data_loader_train
        elif data_type == "val":
            data_loader = self.data_loader_val

        loss_y = 0
        n = 0
        for x_case, x_event, prefix_len, t, y in data_loader:
            loss_y += self.causal_functions.get_loss_y_t_x(x_case=x_case, x_event=x_event, prefix_len=prefix_len, t=t, y=y) * x_event.size(0)
            n += x_event.size(0)
            # 24/01/2025 --> x_event.size(0) is the batch size, so correct to multiply by this
        return (loss_y / n).item()

    # def evaluate(self, data_type):
    #     if data_type == "train":
    #         data_loader = self.data_loader_train
    #     elif data_type == "val":
    #         data_loader = self.data_loader_val

    #     loss_y = 0
    #     loss_t = 0
    #     n = 0
    #     for x_case, x_event, prefix_len, t, y in data_loader:
    #         loss_y += self.causal_functions.get_loss_y_t_x(x_case=x_case, x_event=x_event, prefix_len=prefix_len, t=t, y=y)[0] * x_event.size(0)
    #         loss_t += self.causal_functions.get_loss_t_x(x_case=x_case, x_event=x_event, prefix_len=prefix_len, t=t)[0] * x_event.size(0)
    #         n += x_event.size(0)
    #     return ( loss_y / n).item(), (loss_t / n).item()
    
    def _sample_t(self, x_case, x_event, prefix_len=None, overlap=1):
        t_ = self.causal_functions.forward_t_x(x_case=x_case, x_event=x_event, prefix_len=prefix_len)
        t_indices = self.treatment_distribution.sample(t_, overlap=overlap)
        if self.MODEL_PARAMS["t"]["dim_t"] > 1:
            t_samples = torch.eye(self.MODEL_PARAMS["t"]["dim_t"])[t_indices.long()].squeeze()
        else:
            t_samples = t_indices
        return t_samples

    def _sample_y(self, t, x_case=None, x_event=None, prefix_len=None, ret_counterfactuals=False):
        if self.MODEL_PARAMS["ignore_x"]:
            x_case = torch.zeros_like(x_case)
            x_event = torch.zeros_like(x_event)
        
        if ret_counterfactuals:
            y_total_ = self.causal_functions.forward_y_t_x(x_case=x_case, x_event=x_event, prefix_len=prefix_len, t=t, ret_counterfactuals=True)
            y_samples_total = [torch.tensor(self.outcome_distribution.sample(y_)) for y_ in y_total_]
            return y_samples_total
            # y0_, y1_ = self.causal_functions.forward_y_t_x(x_case=x_case, x_event=x_event, prefix_len=prefix_len, t=t, ret_counterfactuals=True)
            # y0_samples = self.outcome_distribution.sample(y0_)
            # y1_samples = self.outcome_distribution.sample(y1_)
            # return y0_samples, y1_samples
        else:
            y_ = self.causal_functions.forward_y_t_x(x_case=x_case, x_event=x_event, prefix_len=prefix_len, t=t, ret_counterfactuals=False)
            y_samples = self.outcome_distribution.sample(y_)
            return y_samples

    def mean_y(self, t, x_case, x_event, prefix_len=None):
        if self.MODEL_PARAMS["ignore_x"]:
            x_case = torch.zeros_like(x_case)
            x_event = torch.zeros_like(x_event)
        return self.outcome_distribution.mean(self.causal_functions.forward_y_t_x(x_case=x_case, x_event=x_event, prefix_len=prefix_len, t=t))
    
    def val(self, t_only=False, y_only=False):
        # Set in evaluation mode
        for net in self.networks_y: net.eval()
        self.network_t.eval()

        uni_metrics_val = self.get_univariate_quant_metrics(dataset="val", verbose=False, outcome_distribution=self.outcome_distribution,t_only=t_only, y_only=y_only)

        print('\n')
        print("Univariate metrics val: ", uni_metrics_val)

        for net in self.networks_y: net.train()
        self.network_t.train()
        return uni_metrics_val
    
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
        # return multi_metrics_test