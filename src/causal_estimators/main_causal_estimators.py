from src.causal_estimators.s_learner import SLearnerFunctions
from src.causal_estimators.tarnet import TarNetFunctions
from src.causal_estimators.t_learner import TLearnerFunctions
import numpy as np
import torch
from torch.utils import data
from itertools import chain
from tqdm import tqdm
from contextlib import contextmanager

@contextmanager
def eval_ctx(mdl, debug=False, is_train=False):
    for net in mdl.networks: net.eval()
    torch.autograd.set_detect_anomaly(debug)
    with torch.set_grad_enabled(mode=is_train):
        yield
    torch.autograd.set_detect_anomaly(False)
    for net in mdl.networks: net.train()

class SequentialDataset(data.Dataset):
    def __init__(self, x_case, x_event, prefix_len, t, y, xtype='float32', ttype='float32', ytype='float32', case_nrs=None):
        self.x_case = x_case.to(torch.float32)
        self.x_event = x_event.to(torch.float32)
        self.prefix_len = prefix_len
        self.t = t.to(torch.float32)
        self.y = y.to(torch.float32)
        if case_nrs is not None:
            self.case_nrs = case_nrs.to(torch.float32)

    def __len__(self):
        return self.x_event.size(0)

    def __getitem__(self, index):
        return (
            self.x_case[index],
            self.x_event[index],
            self.prefix_len[index],
            self.t[index],
            self.y[index]
        ) if not hasattr(self, 'case_nrs') else (
            self.x_case[index],
            self.x_event[index],
            self.prefix_len[index],
            self.t[index],
            self.y[index],
            self.case_nrs[index]
        )

class AggregatedDataset(data.Dataset):
    def __init__(self, x, t, y, xtype='float32', ttype='float32', ytype='float32', case_nrs=None):
        self.x = x.to(torch.float32)
        self.t = t.to(torch.float32)
        self.y = y.to(torch.float32)
        if case_nrs is not None:
            self.case_nrs = case_nrs.to(torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return (
            self.x[index],
            self.t[index],
            self.y[index]
        ) if not hasattr(self, 'case_nrs') else (
            self.x[index],
            self.t[index],
            self.y[index],
            self.case_nrs[index]
        )
    
class CausalEstimator():
    def __init__(self, data_train, data_val, data_val_th, prep_utils, MODEL_PARAMS, inference=False):
        self.data_train = data_train
        self.data_val = data_val
        self.data_val_th = data_val_th
        self.prep_utils = prep_utils
        self.MODEL_PARAMS = MODEL_PARAMS
        self.inference = inference
        # Set seed
        torch.manual_seed(MODEL_PARAMS["seed"])
        torch.cuda.manual_seed(MODEL_PARAMS["seed"])
        torch.cuda.manual_seed_all(MODEL_PARAMS["seed"])

        if MODEL_PARAMS["model_type"] in ["LSTM", "CNN", "LSTM-VAE"]:
            self.dim_x_case = len(self.prep_utils["case_cols_encoded"])
            self.dim_x_event = len(self.prep_utils["event_cols_encoded"])
            self.dim_t = self.MODEL_PARAMS["dim_t"]
            self.dim_y = self.MODEL_PARAMS["dim_y"]
            self.dim_hidden_lstm = self.MODEL_PARAMS["dim_hidden_lstm"]
            self.dim_hidden_dense = self.MODEL_PARAMS["dim_hidden_dense"]
            self.dims = {"dim_x_case": self.dim_x_case, "dim_x_event": self.dim_x_event, "dim_t": self.dim_t, "dim_y": self.dim_y, "dim_hidden_lstm": self.dim_hidden_lstm, "dim_hidden_dense": self.dim_hidden_dense}

            if not self.inference:
                self.data_loader_train = data.DataLoader(
                    SequentialDataset(x_case=self.data_train["X_case"], x_event=self.data_train["X_event"], prefix_len=self.data_train["prefix_len"], t=self.data_train["T"], y=self.data_train["Y"]),
                    batch_size=self.MODEL_PARAMS["batch_size"],
                    shuffle=True,
                )

                self.data_loader_val = data.DataLoader(
                    SequentialDataset(x_case=self.data_val["X_case"], x_event=self.data_val["X_event"], prefix_len=self.data_val["prefix_len"], t=self.data_val["T"], y=self.data_val["Y"]),
                    batch_size=self.MODEL_PARAMS["batch_size"],
                    shuffle=True,
                )

                # NOTE: this is only used for threshold tuning, batch size is 1 to loop over all prefixes, and shuffle is False to keep the order
                self.data_loader_val_th = data.DataLoader(
                    SequentialDataset(x_case=self.data_val_th["X_case"], x_event=self.data_val_th["X_event"], prefix_len=self.data_val_th["prefix_len"], t=self.data_val_th["T"], y=self.data_val_th["Y"], case_nrs=self.data_val_th["case_nr"]),
                    batch_size=1,
                    shuffle=False,
                )
        else:
            # check if "X" is in data_train, if not, use the prep_utils to get the column names
            if "X" not in self.data_train:
                self.dim_x = len(self.prep_utils["column_names"]) - 1 - 1 - 1 # -1 for the outcome, -1 for the treatment, -1 for the case_nr
            else:
                self.dim_x = self.data_train["X"].shape[1]
            # self.dim_x = len(self.prep_utils["column_names"]) - 1 - 1 # -1 for the outcome, -1 for the treatment
            self.dim_t = self.MODEL_PARAMS["dim_t"]
            self.dim_y = self.MODEL_PARAMS["dim_y"]
            self.dim_hidden_dense = self.MODEL_PARAMS["dim_hidden_dense"]
            self.dims = {"dim_x": self.dim_x, "dim_t": self.dim_t, "dim_y": self.dim_y, "dim_hidden_dense": self.dim_hidden_dense}

            if not self.inference:
                self.data_loader_train = data.DataLoader(
                    AggregatedDataset(x=self.data_train["X"], t=self.data_train["T"], y=self.data_train["Y"]),
                    batch_size=self.MODEL_PARAMS["batch_size"],
                    shuffle=True,
                )

                self.data_loader_val = data.DataLoader(
                    AggregatedDataset(x=self.data_val["X"], t=self.data_val["T"], y=self.data_val["Y"]),
                    batch_size=self.MODEL_PARAMS["batch_size"],
                    shuffle=True,
                )

                # NOTE: this is only used for threshold tuning, batch size is 1 to loop over all prefixes, and shuffle is False to keep the order
                self.data_loader_val_th = data.DataLoader(
                    AggregatedDataset(x=self.data_val_th["X"], t=self.data_val_th["T"], y=self.data_val_th["Y"], case_nrs=self.data_val_th["case_nr"]),
                    batch_size=1,
                    shuffle=False,
                )

        if MODEL_PARAMS["causal_type"] == "T-Learner":
            self.causal_functions = TLearnerFunctions(dims=self.dims, MODEL_PARAMS=MODEL_PARAMS)
        elif MODEL_PARAMS["causal_type"] == "S-Learner":
            self.causal_functions = SLearnerFunctions(dims=self.dims, MODEL_PARAMS=MODEL_PARAMS)
        elif MODEL_PARAMS["causal_type"] == "TarNet":
            self.causal_functions = TarNetFunctions(dims=self.dims, MODEL_PARAMS=MODEL_PARAMS)
        
        self.networks = self.causal_functions.networks

        self.optim = torch.optim.Adam(
            chain(*[net.parameters() for net in self.networks]),
            self.MODEL_PARAMS["lr"]
        )

    def train(self):
        if self.MODEL_PARAMS["model_type"] in ["LSTM", "CNN", "LSTM-VAE"]:
            self.train_seq()
        else:
            self.train_agg()

    """Train a model that uses aggregation encoding."""
    def train_agg(self):
        self.val_losses = []
        self.losses = []
        loss_val = float("inf")

        c = 0
        self.best_val_loss = float("inf")
        self.best_val_idx = 0
        for _ in tqdm(range(self.MODEL_PARAMS["num_epochs"]), disable=self.MODEL_PARAMS["disable_progress_bar"]):
            for x, t, y in self.data_loader_train:
                self.optim.zero_grad()
                loss = self.causal_functions.get_loss_agg(x=x, t=t, y=y)
                self.losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(chain(*[net.parameters() for net in self.networks]), self.MODEL_PARAMS["grad_norm"])
                self.optim.step()

                c += 1
                if c % self.MODEL_PARAMS["print_every_iters"] == 0:
                    print("\n")
                    print("Iteration :", c)
                    print('    Training loss:', loss.item())

                if c % self.MODEL_PARAMS["eval_every"] == 0:
                    with eval_ctx(self):
                        loss_val = self.evaluate_agg(data_type="val")
                    self.val_losses.append(loss_val)
                    print("    Val loss:", loss_val)
                    if loss_val < self.best_val_loss:
                        self.best_val_loss = loss_val
                        self.best_val_idx = c
                        print("    saving best-val-loss model")
                        torch.save([net.state_dict() for net in self.networks], self.MODEL_PARAMS["savepath"])

            if self.MODEL_PARAMS["early_stop"] and self.MODEL_PARAMS["patience"] is not None and c - self.best_val_idx > self.MODEL_PARAMS["patience"]:
                print('early stopping criterion reached. Ending experiment. ')
                break
        
        if self.MODEL_PARAMS["early_stop"]:
            print("loading best-val-loss model (early stopping checkpoint)")
            for net, params in zip(self.networks, torch.load(self.MODEL_PARAMS["savepath"])):
                # put in eval mode
                net.eval()
                net.load_state_dict(params)

    def evaluate_agg(self, data_type):
        if data_type == "val":
            data_loader = self.data_loader_val
        else:
            raise ValueError("data_type must be'val'")

        losses = []
        for x, t, y in data_loader:
            with eval_ctx(self, is_train=False):
                loss = self.causal_functions.get_loss_agg(x=x, t=t, y=y)
                losses.append(loss.item())
        return np.mean(losses)
    
    """Train a model that uses sequential encoding."""
    def train_seq(self):
        self.val_losses = []
        self.losses = []
        loss_val = float("inf")

        c = 0
        self.best_val_loss = float("inf")
        self.best_val_idx = 0
        for _ in tqdm(range(self.MODEL_PARAMS["num_epochs"]), disable=self.MODEL_PARAMS["disable_progress_bar"]):
            for x_case, x_event, prefix_len, t, y in self.data_loader_train:
                self.optim.zero_grad()
                loss = self.causal_functions.get_loss_seq(x_case=x_case, x_event=x_event, prefix_len=prefix_len, t=t, y=y)
                self.losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(chain(*[net.parameters() for net in self.networks]), self.MODEL_PARAMS["grad_norm"])
                self.optim.step()

                c += 1
                if c % self.MODEL_PARAMS["print_every_iters"] == 0:
                    print("\n")
                    print("Iteration :", c)
                    print('    Training loss:', loss.item())

                if c % self.MODEL_PARAMS["eval_every"] == 0:
                    with eval_ctx(self):
                        loss_val = self.evaluate_seq(data_type="val")
                    self.val_losses.append(loss_val)
                    print("    Val loss:", loss_val)
                    if loss_val < self.best_val_loss:
                        self.best_val_loss = loss_val
                        self.best_val_idx = c
                        print("    saving best-val-loss model")
                        torch.save([net.state_dict() for net in self.networks], self.MODEL_PARAMS["savepath"])

            if self.MODEL_PARAMS["early_stop"] and self.MODEL_PARAMS["patience"] is not None and c - self.best_val_idx > self.MODEL_PARAMS["patience"]:
                print('early stopping criterion reached. Ending experiment. ')
                break
        
        if self.MODEL_PARAMS["early_stop"]:
            print("loading best-val-loss model (early stopping checkpoint)")
            for net, params in zip(self.networks, torch.load(self.MODEL_PARAMS["savepath"])):
                # put in eval mode
                net.eval()
                net.load_state_dict(params)
    
    def evaluate_seq(self, data_type):
        if data_type == "val":
            data_loader = self.data_loader_val
        else:
            raise ValueError("data_type must be 'val'")

        losses = []
        for x_case, x_event, prefix_len, t, y in data_loader:
            with eval_ctx(self, is_train=False):
                loss = self.causal_functions.get_loss_seq(x_case=x_case, x_event=x_event, prefix_len=prefix_len, t=t, y=y)
                losses.append(loss.item())
        return np.mean(losses)
    
    def tune_threshold(self, nr_thresholds=5):
        # put networks in eval mode
        for net in self.networks:
            net.eval()
        for net in self.causal_functions.networks:
            net.eval()
        
        if self.MODEL_PARAMS["model_type"] in ["LSTM", "CNN", "LSTM-VAE"]:
            th_func = self.get_threshold_perf_seq
        else:
            th_func = self.get_threshold_perf_agg
        
        # get the perf of th = 0, and the min and max diff_y
        sum_th_0, min_diff_y, max_diff_y = th_func(0, get_min_max=True)

        # from 0 to max_diff_y, split to get 4 th's
        # th_values = np.linspace(0, max_diff_y.item(), 5)
        # th_values = th_values[1:]
        th_values = np.linspace(max(min_diff_y.item(), 0), max_diff_y.item(), nr_thresholds)
        best_th = 0
        best_sum = sum_th_0
        for th in th_values:
            sum = th_func(th)
            if sum > best_sum:
                best_sum = sum
                best_th = th
        return best_th, best_sum
    
    def get_threshold_perf_agg(self, th, get_min_max=False):
        # NOTE: only for an intervention with a single treatment

        # given an array of thresholds
        # get the best th by going over the val_th data
        # for each case, go through the prefixes, and get the CATE (predict y under each treatment)
        # if the CATE > the, then get y1, if not, go to the next prefix of the case. If the treatment occured in the historical data, just get y0
        # save the estimated y for every case, and sum them up
        # the best th is the one that maximizes the sum of the estimated y
        current_case_nr = None
        # Remember the sum for each threshold
        sum = 0
        max_diff_y = -float("inf")
        min_diff_y = float("inf")
        y_to_sum = 0
        for x, t, y, case_nr in self.data_loader_val_th:
            if current_case_nr is None:
                current_case_nr = case_nr
                policy_treated_case = False
                historical_treated_case = False

            # y_to_sum = None
            if case_nr != current_case_nr:
                # if the estimator never recommended the treatment
                if not policy_treated_case:
                    # if the treatment was never recommended in the case, but it was done in the historical data, get the estimated y0
                    if historical_treated_case:
                        y_to_sum = y_total[0]
                    # but if the treatment was never recommended in the case, and it was NOT done in the historical data, get real y
                    else:
                        y_to_sum = y
                # sum it up for the current th
                sum += y_to_sum.sum()
                current_case_nr = case_nr
                historical_treated_case = False
                policy_treated_case = False
            elif policy_treated_case:
                continue
            
            # get the estimated y's for the current prefix
            with eval_ctx(self, is_train=False):
                y_total = self.causal_functions.forward_agg(x=x, t=t, ret_counterfactuals=True)
            # get the CATE
            diff_y = y_total[1] - y_total[0]

            if diff_y > max_diff_y:
                max_diff_y = diff_y
            if diff_y < min_diff_y:
                min_diff_y = diff_y

            if diff_y > th:
                # if the treatment is recommended, and it was done at this moment in the historical data, get the real y (please note that t is a tensor, so just check if it contains 1)
                if t.item() == 1:
                    y_to_sum = y
                # if the treatment is recommended, and it was NOT done at this moment in the historical data, get the estimated y1
                else:
                    y_to_sum = y_total[1]
                policy_treated_case = True

            if t.item() == 1:
                historical_treated_case = True

        if get_min_max:
            return sum, min_diff_y, max_diff_y
        return sum
    
    def get_threshold_perf_seq(self, th, get_min_max=False):
        # NOTE: only for an intervention with a single treatment

        # given an array of thresholds
        # get the best th by going over the val_th data
        # for each case, go through the prefixes, and get the CATE (predict y under each treatment)
        # if the CATE > th, then get y1, if not, go to the next prefix of the case. If the treatment occured in the historical data, just get y0
        # save the estimated y for every case, and sum them up
        # the best th is the one that maximizes the sum of the estimated y
        current_case_nr = None
        # Remember the sum for each threshold
        sum = 0
        max_diff_y = -float("inf")
        min_diff_y = float("inf")
        y_to_sum = 0
        for x_case, x_event, prefix_len, t, y, case_nr in self.data_loader_val_th:
            if current_case_nr is None:
                current_case_nr = case_nr
                policy_treated_case = False
                historical_treated_case = False

            # y_to_sum = None
            if case_nr != current_case_nr:
                # if the estimator never recommended the treatment
                if not policy_treated_case:
                    # if the treatment was never recommended in the case, but it was done in the historical data, get the estimated y0
                    if historical_treated_case:
                        y_to_sum = y_total[0]
                    # but if the treatment was never recommended in the case, and it was NOT done in the historical data, get real y
                    else:
                        y_to_sum = y
                # sum it up for the current th
                sum += y_to_sum.sum()
                current_case_nr = case_nr
                historical_treated_case = False
                policy_treated_case = False
            elif policy_treated_case:
                continue
            
            # get the estimated y's for the current prefix
            with eval_ctx(self, is_train=False):
                y_total = self.causal_functions.forward_seq(x_case=x_case, x_event=x_event, prefix_len=prefix_len, t=t, ret_counterfactuals=True)
            # get the CATE
            diff_y = y_total[1] - y_total[0]

            if diff_y > max_diff_y:
                max_diff_y = diff_y
            if diff_y < min_diff_y:
                min_diff_y = diff_y

            if diff_y > th:
                # if the treatment is recommended, and it was done at this moment in the historical data, get the real y (please note that t is a tensor, so just check if it contains 1)
                if t.item() == 1:
                    y_to_sum = y
                # if the treatment is recommended, and it was NOT done at this moment in the historical data, get the estimated y1
                else:
                    y_to_sum = y_total[1]
                policy_treated_case = True

            if t.item() == 1:
                historical_treated_case = True
        
        if get_min_max:
            return sum, min_diff_y, max_diff_y
        return sum