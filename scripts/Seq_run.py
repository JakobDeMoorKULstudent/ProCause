DATA_FOLDER = "data"
RESULTS_FOLDER = "res"
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from config.config import path
sys.path.append(path)

from src.utils.tools import save_data, load_data
from copy import deepcopy
import torch
from src.utils.preprocessor.main_prep import ProcessPreprocessor
from Seq.seq_generator import SeqGenerator
from src.utils.simbank_eval.main_eval import SimBankEvaluator
from src.utils.hp_tuning import HyperParamTuner
from hyperopt import hp, space_eval
from SimBank.confounding_level import set_delta
import numpy as np
import argparse
import pandas as pd
import json

parser = argparse.ArgumentParser(description='ProCause')
parser.add_argument('--config', type=str, help='Path to config file')
args, unknown = parser.parse_known_args()
config_args = {}
if args.config:
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_args = json.load(f)

parser.add_argument('--dataset', type=str, default=config_args.get('dataset', "SimBank"), help='Dataset')
parser.add_argument('--intervention_name', nargs='+', type=str, default=config_args.get('intervention_name', ["set_ir_3_levels"]), help='Intervention name')
parser.add_argument('--learners', nargs='+', type=str, default=config_args.get('learners', ["TarNet", "T-Learner", "S-Learner"]), help='Learners')
parser.add_argument('--big_data', type=lambda x: x.lower() == 'true', default=config_args.get('big_data', True), help='Big data (True or False)')
parser.add_argument('--big_eval', type=lambda x: x.lower() == 'true', default=config_args.get('big_eval', True), help='Big eval (True or False)')
parser.add_argument('--t_already_trained', type=lambda x: x.lower() == 'true', default=config_args.get('t_already_trained', False), help='T already trained (True or False)')
parser.add_argument('--y_already_trained', type=lambda x: x.lower() == 'true', default=config_args.get('y_already_trained', False), help='Y already trained (True or False)')
parser.add_argument('--already_trained', type=lambda x: x.lower() == 'true', default=config_args.get('already_trained', False), help='Already trained (True or False)')
parser.add_argument('--already_preprocessed', type=lambda x: x.lower() == 'true', default=config_args.get('already_preprocessed', False), help='Already preprocessed (True or False)')
parser.add_argument('--iterations_to_skip', nargs='+', type=int, default=config_args.get('iterations_to_skip', []), help='Iterations to skip')
parser.add_argument('--biased', type=lambda x: x.lower() == 'true', default=config_args.get('biased', True), help='Biased (True or False)')
parser.add_argument('--tuning', type=lambda x: x.lower() == 'true', default=config_args.get('tuning', False), help='Tuning (True or False)')
parser.add_argument('--num_iterations', type=int, default=config_args.get('num_iterations', 10), help='Num iterations')
parser.add_argument('--delta', type=float, default=config_args.get('delta', 0.95), help='Delta')
parser.add_argument('--t_already_tuned', type=lambda x: x.lower() == 'true', default=config_args.get('t_already_tuned', False), help='T already tuned (True or False)')
parser.add_argument('--y_already_tuned', type=lambda x: x.lower() == 'true', default=config_args.get('y_already_tuned', False), help='Y already tuned (True or False)')
parser.add_argument('--vsc', type=lambda x: x.lower() == 'true', default=config_args.get('vsc', True), help='VSC (True or False)')
parser.add_argument('--logging', type=lambda x: x.lower() == 'true', default=config_args.get('logging', True), help='Logging (True or False)')
parser.add_argument('--without_cum_cost', type=lambda x: x.lower() == 'true', default=config_args.get('without_cum_cost', True), help='Without cum cost (True or False)')
parser.add_argument('--testing', type=lambda x: x.lower() == 'true', default=config_args.get('testing', False), help='Testing (True or False)')
parser.add_argument('--just_take_best', type=lambda x: x.lower() == 'true', default=config_args.get('just_take_best', False), help='Just take best (True or False)')
parser.add_argument('--policies', nargs='+', type=str, default=config_args.get('policies', ["all"]), help='Evaluation policies')
parser.add_argument('--get_dfs_policies', type=lambda x: x.lower() == 'true', default=config_args.get('get_dfs_policies', False), help='Get dfs policies (True or False)')
parser.add_argument('--get_dfs_policies_only', type=lambda x: x.lower() == 'true', default=config_args.get('get_dfs_policies_only', False), help='Get dfs policies only (True or False)')
parser.add_argument('--fixed_policy_delta', type=float, default=config_args.get('fixed_policy_delta', None), help='Fixed policy delta')

args = parser.parse_args()

print(args)

args.just_take_best_STRING = ""
if args.just_take_best:
    args.iterations_to_skip = [1, 2, 3, 4]
    args.just_take_best_STRING = "_just_take_best"
    args.already_trained = True

# MAIN ARGUMENTS
DATASET = args.dataset
DATASET_PARAMS = {}
STAT_TESTS = False

PREP_PARAMS = {"test_prop": 0, "val_prop": 0.2, "standardize": True, "one_hot_encode": True, "impute": True, "seed": 0, "encoding": "sequential"} 
MODEL_PARAMS = {"early_stop": True, "ignore_x": False, "print_every_iters": 100000000000, "eval_every": 2000, "plot_every": 10000000000, "p_every": 10000000000, 
                "num_epochs": 100000, "masked": True, "disable_progress_bar": True}
TUNING_PARAMS_SPACE = {
    'lr': hp.loguniform('lr', np.log(0.00001), np.log(0.001)),  # Learning rate in log scale
    'batch_size': hp.choice('batch_size', [64, 256, 1024]),
    "dim_hidden": hp.choice('dim_hidden', [32, 64]),
}

if DATASET == "SimBank":
    DATASET_PARAMS["intervention_name"] = args.intervention_name
    DATASET_PARAMS["train_size"] = 10000
    PATH_SUFFIX_DATA = "ProCause_training"
    dataset_params_simbank = load_data(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "loan_log_" + str(DATASET_PARAMS["intervention_name"]) + "_" + str(DATASET_PARAMS["train_size"]) + "_dataset_params_" + PATH_SUFFIX_DATA))
    DATASET_PARAMS.update(dataset_params_simbank)
    if args.biased:
        DATASET_PARAMS["data_path_normal"] = "loan_log_" + str(DATASET_PARAMS["intervention_name"]) + "_" + str(DATASET_PARAMS["train_size"]) + "_train_normal_" + PATH_SUFFIX_DATA
        BIAS_PATH = "_biased"
        # LOAD DATA
        data_normal = load_data(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, DATASET_PARAMS["data_path_normal"]))
        DATASET_PARAMS["data_path_RCT"] = "loan_log_" + str(DATASET_PARAMS["intervention_name"]) + "_" + str(DATASET_PARAMS["train_size"]) + "_train_RCT_" + PATH_SUFFIX_DATA
        data_RCT = load_data(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, DATASET_PARAMS["data_path_RCT"]))
        delta = args.delta
        if delta != 0.95:
            BIAS_PATH = "_biased_" + str(delta)

        # cut the data here already based on train size (is normally 10 000 so will be fine)
        case_nrs_normal = data_normal["case_nr"].unique()
        case_nrs_normal = case_nrs_normal[:DATASET_PARAMS["train_size"]]
        data = data_normal[data_normal["case_nr"].isin(case_nrs_normal)]

        case_nrs_RCT = data_RCT["case_nr"].unique()
        case_nrs_RCT = case_nrs_RCT[:DATASET_PARAMS["train_size"]]
        data_RCT = data_RCT[data_RCT["case_nr"].isin(case_nrs_RCT)]

        data = set_delta(data=data_normal, data_RCT=data_RCT, delta=delta)
    else:
        DATASET_PARAMS["data_path_RCT"] = "loan_log_" + str(DATASET_PARAMS["intervention_name"]) + "_" + str(DATASET_PARAMS["train_size"]) + "_train_RCT_" + PATH_SUFFIX_DATA
        BIAS_PATH = ""
        data = load_data(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, DATASET_PARAMS["data_path_RCT"]))

        case_nrs = data["case_nr"].unique()
        case_nrs = case_nrs[:DATASET_PARAMS["train_size"]]
        data = data[data["case_nr"].isin(case_nrs)]

    DATASET_PARAMS["case_nr_column"] = "case_nr"
    DATASET_PARAMS["order_column"] = "timestamp"
    if args.without_cum_cost:
        DATASET_PARAMS["event_cols"].remove("cum_cost")
        DATASET_PARAMS["scale_cols"].remove("cum_cost")
    DATASET_PARAMS["nr_of_interventions"] = len(DATASET_PARAMS["intervention_name"])
    if DATASET_PARAMS["intervention_name"][0] == "set_ir_3_levels":
        DATASET_PARAMS["action_width"] = [3]
        DATASET_PARAMS["action_depth"] = [1]
    elif DATASET_PARAMS["intervention_name"][0] == "time_contact_HQ":
        DATASET_PARAMS["action_width"] = [2]
        DATASET_PARAMS["action_depth"] = [4]

    PREP_PARAMS["calc_atoms"] = True
    PREP_PARAMS["group_size_atoms"] = 5
    PREP_PARAMS["filter_useless_cols"] = True

    MODEL_PARAMS["t"] = {"lr": 0.0001, "batch_size": 1024, "dim_hidden": 25, "dim_t": 1 if DATASET_PARAMS["action_width"][0] == 2 else DATASET_PARAMS["action_width"][0],
                         "treatment_distribution": ["bernoulli"] if DATASET_PARAMS["action_width"][0] == 2 else ["categorical"]}
    MODEL_PARAMS["y"] = {"lr": 0.0001, "batch_size": 1024, "dim_hidden": 25, "dim_y": 1, "outcome_distribution": ["normal", "atoms"]}

    MODEL_PARAMS["causal_type"] = args.learners[0]
    MODEL_PARAMS["model_type"] = "LSTM"
    MODEL_PARAMS["n_dense_in_lstm"] = 2
    MODEL_PARAMS["n_lstm"] = 2
    MODEL_PARAMS["n_dense"] = 3
    MODEL_PARAMS["dim_sigmoidflow"] = 2
    MODEL_PARAMS["patience"] = 7500
    MODEL_PARAMS["grad_norm"] = 1.0
    MODEL_PARAMS["t_already_trained"] = False
    if MODEL_PARAMS["causal_type"] == "TarNet":
        MODEL_PARAMS["n_lstm_y"] = 2
        MODEL_PARAMS["n_dense_in_lstm_y"] = 1
        MODEL_PARAMS["hidden_size_multiplier_lstm_y"] = 0.65
        MODEL_PARAMS["n_dense_y"] = 2
        MODEL_PARAMS["hidden_size_multiplier_dense_y"] = 2

    if "all" in args.policies:
        EVALUATOR_PARAMS = {"policies": [{"name": "bank"}, {"name": "random"},
                                        {"name": "S-Learner_LSTM", "causal_type": "S-Learner", "model_type": "LSTM"}, 
                                        {"name": "TarNet_LSTM", "causal_type": "TarNet", "model_type": "LSTM"}, 
                                        {"name": "T-Learner_LSTM", "causal_type": "T-Learner", "model_type": "LSTM"},
                                        {"name": "S-Learner_Vanilla_NN", "causal_type": "S-Learner", "model_type": "Vanilla_NN"},
                                        {"name": "TarNet_Vanilla_NN", "causal_type": "TarNet", "model_type": "Vanilla_NN"},
                                        {"name": "T-Learner_Vanilla_NN", "causal_type": "T-Learner", "model_type": "Vanilla_NN"}],
                            "nr_cases": 1000, "nr_samples_per_case": 50, "num_iterations": 5}

    else:
        EVALUATOR_PARAMS = {"policies": [], "nr_cases": 1000, "nr_samples_per_case": 50, "num_iterations": 5}
        for policy in args.policies:
            if policy in ["bank", "random"]:
                EVALUATOR_PARAMS["policies"].append({"name": policy})
            else:
                # for the causal_type grab the first part of the policy name, and for the model_type grab the second part of the policy name
                causal_type = policy.split("_")[0]
                model_type = policy.split("_")[1]
                EVALUATOR_PARAMS["policies"].append({"name": policy, "causal_type": causal_type, "model_type": model_type})
    
    if not args.big_eval:
        EVALUATOR_PARAMS["nr_cases"], EVALUATOR_PARAMS["nr_samples_per_case"], EVALUATOR_PARAMS["num_iterations"] = 50, 5, 2
        if "all" in args.policies:
            EVALUATOR_PARAMS["policies"] = [{"name": "bank"}, {"name": "random"},
                                            {"name": "T-Learner_LSTM", "causal_type": "T-Learner", "model_type": "LSTM"},
                                            {"name": "T-Learner_Vanilla_NN", "causal_type": "T-Learner", "model_type": "Vanilla_NN"}
                                        ]
        else:
            EVALUATOR_PARAMS["policies"] = []
            for policy in args.policies:
                if policy in ["bank", "random"]:
                    EVALUATOR_PARAMS["policies"].append({"name": policy})
                else:
                    causal_type = policy.split("_")[0]
                    model_type = policy.split("_")[1]
                    EVALUATOR_PARAMS["policies"].append({"name": policy, "causal_type": causal_type, "model_type": model_type})

elif DATASET == "bpic2012":
    STAT_TESTS = False
    PREP_PARAMS["test_prop"] = 0.2
    BIAS_PATH = ""
    PATH_SUFFIX_DATA = ""
    DATASET_PARAMS["dataset"] = DATASET
    DATASET_PARAMS["intervention_name"] = "sent_2_offers"
    DATASET_PARAMS["case_nr_column"] = "case_nr"
    DATASET_PARAMS["order_column"] = "timestamp"
    DATASET_PARAMS["event_cols"] = ["activity", "resource", "elapsed_time"]
    DATASET_PARAMS["case_cols"] = ["amount_req"]
    DATASET_PARAMS["scale_cols"] = ["amount_req", "elapsed_time"]
    DATASET_PARAMS["cat_cols"] = ["activity", "resource"]
    DATASET_PARAMS["nr_of_interventions"] = 1
    DATASET_PARAMS["action_width"] = [2]
    DATASET_PARAMS["action_depth"] = ["max length of trace"]
    data = pd.read_csv(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "bpic2012_final.csv"), sep=";")

    PREP_PARAMS["calc_atoms"] = False
    PREP_PARAMS["group_size_atoms"] = 0
    PREP_PARAMS["filter_useless_cols"] = True

    MODEL_PARAMS["t"] = {"lr": 0.0001, "batch_size": 1024, "dim_hidden": 25, "dim_t": 1 if DATASET_PARAMS["action_width"][0] == 2 else DATASET_PARAMS["action_width"][0],
                            "treatment_distribution": ["bernoulli"] if DATASET_PARAMS["action_width"][0] == 2 else ["categorical"]}
    MODEL_PARAMS["y"] = {"lr": 0.0001, "batch_size": 1024, "dim_hidden": 25, "dim_y": 1, "outcome_distribution": ["bernoulli"]}
    MODEL_PARAMS["causal_type"] = args.learners[0]
    MODEL_PARAMS["model_type"] = "LSTM"
    MODEL_PARAMS["n_dense_in_lstm"] = 2
    MODEL_PARAMS["n_lstm"] = 2
    MODEL_PARAMS["n_dense"] = 3
    MODEL_PARAMS["patience"] = 7500
    MODEL_PARAMS["grad_norm"] = 1.0
    MODEL_PARAMS["t_already_trained"] = False
    if MODEL_PARAMS["causal_type"] == "TarNet":
        MODEL_PARAMS["n_lstm_y"] = 2
        MODEL_PARAMS["n_dense_in_lstm_y"] = 1
        MODEL_PARAMS["hidden_size_multiplier_lstm_y"] = 0.65
        MODEL_PARAMS["n_dense_y"] = 2
        MODEL_PARAMS["hidden_size_multiplier_dense_y"] = 2
elif DATASET == "bpic2017":
    STAT_TESTS = False
    DATASET_PARAMS["dataset"] = DATASET
    PREP_PARAMS["test_prop"] = 0.2
    BIAS_PATH = ""
    PATH_SUFFIX_DATA = ""
    DATASET_PARAMS["intervention_name"] = "sent_2_offers"
    DATASET_PARAMS["case_nr_column"] = "case_nr"
    DATASET_PARAMS["order_column"] = "timestamp"
    DATASET_PARAMS["event_cols"] = ["activity", "resource", "elapsed_time", "action"]
    DATASET_PARAMS["case_cols"] = ["case:requestedamount", "case:loangoal", "case:applicationtype", "firstwithdrawalamount", "monthlycost", "creditscore", "offeredamount", "numberofterms"]
    DATASET_PARAMS["scale_cols"] = ["case:requestedamount", "elapsed_time", "firstwithdrawalamount", "monthlycost", "creditscore", "offeredamount", "numberofterms"]
    DATASET_PARAMS["cat_cols"] = ["activity", "resource", "action", "case:loangoal", "case:applicationtype"]
    DATASET_PARAMS["nr_of_interventions"] = 1
    DATASET_PARAMS["action_width"] = [2]
    DATASET_PARAMS["action_depth"] = ["max length of trace"]
    data = pd.read_csv(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "bpic2017_final.csv"), sep=";")

    PREP_PARAMS["calc_atoms"] = False
    PREP_PARAMS["group_size_atoms"] = 0
    PREP_PARAMS["filter_useless_cols"] = True

    MODEL_PARAMS["t"] = {"lr": 0.0001, "batch_size": 1024, "dim_hidden": 25, "dim_t": 1 if DATASET_PARAMS["action_width"][0] == 2 else DATASET_PARAMS["action_width"][0],
                            "treatment_distribution": ["bernoulli"] if DATASET_PARAMS["action_width"][0] == 2 else ["categorical"]}
    MODEL_PARAMS["y"] = {"lr": 0.0001, "batch_size": 1024, "dim_hidden": 25, "dim_y": 1, "outcome_distribution": ["bernoulli"]}
    MODEL_PARAMS["causal_type"] = args.learners[0]
    MODEL_PARAMS["model_type"] = "LSTM"
    MODEL_PARAMS["n_dense_in_lstm"] = 2
    MODEL_PARAMS["n_lstm"] = 2
    MODEL_PARAMS["n_dense"] = 3
    MODEL_PARAMS["patience"] = 7500
    MODEL_PARAMS["grad_norm"] = 1.0
    MODEL_PARAMS["t_already_trained"] = False
    if MODEL_PARAMS["causal_type"] == "TarNet":
        MODEL_PARAMS["n_lstm_y"] = 2
        MODEL_PARAMS["n_dense_in_lstm_y"] = 1
        MODEL_PARAMS["hidden_size_multiplier_lstm_y"] = 0.65
        MODEL_PARAMS["n_dense_y"] = 2
        MODEL_PARAMS["hidden_size_multiplier_dense_y"] = 2

if not args.big_data:
    data = data.head(5000)
    MODEL_PARAMS["num_epochs"] = 1
    MODEL_PARAMS["patience"] = 10
    MODEL_PARAMS["lr"] = 0.001
    MODEL_PARAMS["print_every_iters"] = 10
    MODEL_PARAMS["eval_every"] = 10
    MODEL_PARAMS["plot_every"] = 100000
    MODEL_PARAMS["p_every"] = 10000000
else:
    save_data(DATASET_PARAMS, os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "dataset_params_" + str(DATASET_PARAMS["intervention_name"]) + BIAS_PATH + PATH_SUFFIX_DATA))
    save_data(PREP_PARAMS, os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "prep_params_" + str(DATASET_PARAMS["intervention_name"]) + BIAS_PATH + PATH_SUFFIX_DATA))


# PREPROCESS DATA (for each intervention)
data_train_prep_list = []
data_val_prep_list = []
data_test_prep_list = []
prep_utils_list = []

DATASET_PARAMS_LIST = []
if DATASET == "SimBank":
    for intervention in range(DATASET_PARAMS["nr_of_interventions"]):
        params = deepcopy(DATASET_PARAMS)
        for key, value in params["intervention_info"].items():
            if isinstance(value, list):
                params["intervention_info"][key] = value[intervention]
        DATASET_PARAMS_LIST.append(params)
else:
    DATASET_PARAMS_LIST.append(DATASET_PARAMS)

print("\nPREPROCESSING\n")
for intervention in range(DATASET_PARAMS["nr_of_interventions"]):
    if not args.already_preprocessed:
        preprocessor = ProcessPreprocessor(DATASET = DATASET, raw_data = data, PREP_PARAMS = PREP_PARAMS, DATASET_PARAMS = DATASET_PARAMS_LIST[intervention])
        data_train_prep, data_val_prep, data_test_prep, prep_utils = preprocessor.preprocess()
        data_train_prep_list.append(data_train_prep)
        data_val_prep_list.append(data_val_prep)
        data_test_prep_list.append(data_test_prep)
        prep_utils_list.append(prep_utils)

        if args.big_data:
            save_data(data_train_prep, os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "data_train_prep_" + str(DATASET_PARAMS["intervention_name"]) + BIAS_PATH + PATH_SUFFIX_DATA))
            save_data(data_val_prep, os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "data_val_prep_" + str(DATASET_PARAMS["intervention_name"]) + BIAS_PATH + PATH_SUFFIX_DATA))
            save_data(data_test_prep, os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "data_test_prep_" + str(DATASET_PARAMS["intervention_name"]) + BIAS_PATH + PATH_SUFFIX_DATA))
            save_data(prep_utils, os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "prep_utils_" + str(DATASET_PARAMS["intervention_name"]) + BIAS_PATH + PATH_SUFFIX_DATA))
    else:
        data_train_prep = load_data(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "data_train_prep_" + str(DATASET_PARAMS["intervention_name"]) + BIAS_PATH + PATH_SUFFIX_DATA))
        data_val_prep = load_data(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "data_val_prep_" + str(DATASET_PARAMS["intervention_name"]) + BIAS_PATH + PATH_SUFFIX_DATA))
        data_test_prep = load_data(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "data_test_prep_" + str(DATASET_PARAMS["intervention_name"]) + BIAS_PATH + PATH_SUFFIX_DATA))
        prep_utils = load_data(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "prep_utils_" + str(DATASET_PARAMS["intervention_name"]) + BIAS_PATH + PATH_SUFFIX_DATA))
        
        data_train_prep_list.append(data_train_prep)
        data_val_prep_list.append(data_val_prep)
        data_test_prep_list.append(data_test_prep)
        prep_utils_list.append(prep_utils)


if args.tuning:
    for intervention in range(DATASET_PARAMS["nr_of_interventions"]):
        MODEL_PARAMS["seed"] = 81
        if not args.vsc:
            MODEL_PARAMS["savepath_y"] = "C:/Users/u0166838/Music/cache_best_model_y_TUNING_" + str(DATASET_PARAMS["intervention_name"]) + BIAS_PATH
            MODEL_PARAMS["savepath_t"] = "C:/Users/u0166838/Music/cache_best_model_t_TUNING" + str(DATASET_PARAMS["intervention_name"]) + BIAS_PATH + ".pt"
        else:
            MODEL_PARAMS["savepath_y"] = os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "cache_best_model_y_TUNING_" + str(DATASET_PARAMS["intervention_name"]) + "_VSC" + BIAS_PATH)
            MODEL_PARAMS["savepath_t"] = os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "cache_best_model_t_TUNING_" + str(DATASET_PARAMS["intervention_name"]) + "_VSC" + BIAS_PATH + ".pt")
        print('\nTUNING T for Intervention', intervention, "\n")

        data_train_tuning = {}
        data_val_tuning = {}
        data_test_tuning = {}
        for key, tensor in data_val_prep_list[intervention].items():
            split_index = int(len(tensor) * ( 1 - PREP_PARAMS["val_prop"])) # 0.8
            data_train_tuning[key] = tensor[:split_index]
            data_val_tuning[key] = tensor[split_index:]

        tuner = HyperParamTuner(MODEL_PARAMS=MODEL_PARAMS,
                        TUNING_PARAMS_SPACE=TUNING_PARAMS_SPACE,
                        data={"train": data_train_tuning, "val": data_val_tuning, "test": data_test_tuning, "prep_utils": prep_utils_list[intervention]},
                        generator="ProCause", BIG_DATA=args.big_data, VSC=args.vsc, LOGGING=args.logging, STAT_TESTS=STAT_TESTS)

        if not args.t_already_tuned:
            best_params_t, best_network_t = tuner.tune_t()
            if args.big_data:
                save_data(best_params_t, os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "best_params_t_tuning_" + str(DATASET_PARAMS["intervention_name"]) + "TarNet" + BIAS_PATH))
                torch.save(best_network_t.state_dict(), os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "best_model_t_tuning_" + str(DATASET_PARAMS["intervention_name"]) + "TarNet" + BIAS_PATH))

        if not args.y_already_tuned:
            for learner in args.learners:
                MODEL_PARAMS["causal_type"] = learner

                print('\nTUNING Y for Intervention', intervention, "Learner", learner, "\n")
                tuner.MODEL_PARAMS["causal_type"] = learner
                if learner == "TarNet":
                    tuner.MODEL_PARAMS["n_lstm_y"] = 2
                    tuner.MODEL_PARAMS["n_dense_in_lstm_y"] = 1
                    tuner.MODEL_PARAMS["hidden_size_multiplier_lstm_y"] = 0.65
                    tuner.MODEL_PARAMS["n_dense_y"] = 2
                    tuner.MODEL_PARAMS["hidden_size_multiplier_dense_y"] = 2
                tuner.MODEL_PARAMS["savepath_y"] += "_" + tuner.MODEL_PARAMS["causal_type"] + ".pt"
                best_params_y, best_networks_y = tuner.tune_y()
                if args.big_data:
                    save_data(best_params_y, os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "best_params_y_tuning_" + str(DATASET_PARAMS["intervention_name"]) + MODEL_PARAMS["causal_type"] + BIAS_PATH))
                    torch.save([net.state_dict() for net in best_networks_y], os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "best_model_y_tuning_" + str(DATASET_PARAMS["intervention_name"]) + MODEL_PARAMS["causal_type"] + BIAS_PATH))
else:
    for learner in args.learners:
        MODEL_PARAMS["causal_type"] = learner
        if learner == "TarNet":
            MODEL_PARAMS["n_lstm_y"] = 2
            MODEL_PARAMS["n_dense_in_lstm_y"] = 1
            MODEL_PARAMS["hidden_size_multiplier_lstm_y"] = 0.65
            MODEL_PARAMS["n_dense_y"] = 2
            MODEL_PARAMS["hidden_size_multiplier_dense_y"] = 2

        # T best params is always TarNet currently (it is the same for all causal types)
        if not args.get_dfs_policies and not args.get_dfs_policies_only:
            best_params_t = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "best_params_t_tuning_" + str(DATASET_PARAMS["intervention_name"]) + "TarNet" + BIAS_PATH))
            best_params_t = space_eval(TUNING_PARAMS_SPACE, best_params_t)
            best_params_y = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "best_params_y_tuning_" + str(DATASET_PARAMS["intervention_name"]) + MODEL_PARAMS["causal_type"] + BIAS_PATH))
            best_params_y = space_eval(TUNING_PARAMS_SPACE, best_params_y)
            MODEL_PARAMS["t"].update(best_params_t)
            MODEL_PARAMS["y"].update(best_params_y)
        for iteration in range(args.num_iterations):
            if iteration in args.iterations_to_skip or (iteration != 0 and args.get_dfs_policies_only):
                print("Skipping iteration", iteration)
                continue
            MODEL_PARAMS["seed"] = 42 + iteration*5
            if not args.vsc:
                MODEL_PARAMS["savepath_y"] = "C:/Users/u0166838/Music/cache_best_model_y" + str(iteration) + str(DATASET_PARAMS["intervention_name"]) + MODEL_PARAMS["causal_type"] + BIAS_PATH + ".pt"
                MODEL_PARAMS["savepath_t"] = "C:/Users/u0166838/Music/cache_best_model_t" + str(iteration) + str(DATASET_PARAMS["intervention_name"]) + MODEL_PARAMS["causal_type"] + BIAS_PATH + ".pt"
            else:
                MODEL_PARAMS["savepath_y"] = os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "cache_best_model_y" + str(iteration) + str(DATASET_PARAMS["intervention_name"]) + MODEL_PARAMS["causal_type"] + "_VSC" + BIAS_PATH + ".pt")
                MODEL_PARAMS["savepath_t"] = os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "cache_best_model_t" + str(iteration) + str(DATASET_PARAMS["intervention_name"]) + "_VSC" + MODEL_PARAMS["causal_type"] + BIAS_PATH + ".pt")

            for intervention in range(DATASET_PARAMS["nr_of_interventions"]):
                print('\nTRAINING for Iteration', iteration, 'Intervention', intervention, "Learner", learner, "\n")
                data_train_prep = data_train_prep_list[intervention]
                data_val_prep = data_val_prep_list[intervention]
                data_test_prep = data_test_prep_list[intervention]
                prep_utils = prep_utils_list[intervention]
                
                procause_generator = SeqGenerator(data_train=data_train_prep, 
                                                        data_val=data_val_prep,
                                                        data_test=data_test_prep, 
                                                        prep_utils=prep_utils, 
                                                        MODEL_PARAMS=MODEL_PARAMS)
                
                
                if args.already_trained:
                    if args.just_take_best:
                        for net, params in zip(procause_generator.networks_y, torch.load(os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "best_model_y_tuning_" + str(DATASET_PARAMS["intervention_name"]) + MODEL_PARAMS["causal_type"] + BIAS_PATH))):
                            net.load_state_dict(params)
                        procause_generator.network_t.load_state_dict(torch.load(os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "best_model_t_tuning_" + str(DATASET_PARAMS["intervention_name"]) + "TarNet" + BIAS_PATH)))
                    else:
                        if not args.get_dfs_policies and not args.get_dfs_policies_only:
                            for net, params in zip(procause_generator.networks_y, torch.load(os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "procause_model_y_" + str(iteration) + "_" + str(DATASET_PARAMS["intervention_name"]) + MODEL_PARAMS["causal_type"] + BIAS_PATH))):
                                net.load_state_dict(params)
                            procause_generator.network_t.load_state_dict(torch.load(os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "procause_model_t_" + str(iteration) + "_" + str(DATASET_PARAMS["intervention_name"]) + "TarNet" + BIAS_PATH)))
                            MODEL_PARAMS = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "procause_model_params_" + str(iteration) + "_" + str(DATASET_PARAMS["intervention_name"]) + MODEL_PARAMS["causal_type"] + BIAS_PATH))
                else:
                    if not args.t_already_trained and learner == "TarNet":
                        # TRAIN MODEL T
                        print('Training T')
                        procause_generator.train_t()

                        # SAVE MODEL T
                        if args.big_data:
                            torch.save(procause_generator.network_t.state_dict(), os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "procause_model_t_" + str(iteration) + "_" + str(DATASET_PARAMS["intervention_name"]) + MODEL_PARAMS["causal_type"] + BIAS_PATH))
                            save_data(MODEL_PARAMS, os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "procause_model_params_" + str(iteration) + "_" + str(DATASET_PARAMS["intervention_name"]) + MODEL_PARAMS["causal_type"] + BIAS_PATH))

                    if not args.y_already_trained:
                        print('Training Y', "\n")
                        # TRAIN MODEL Y
                        procause_generator.train_y()

                        # SAVE MODEL Y
                        if args.big_data:
                            torch.save([net.state_dict() for net in procause_generator.networks_y], os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "procause_model_y_" + str(iteration) + "_" + str(DATASET_PARAMS["intervention_name"]) + MODEL_PARAMS["causal_type"] + BIAS_PATH))
                            save_data(MODEL_PARAMS, os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "procause_model_params_" + str(iteration) + "_" + str(DATASET_PARAMS["intervention_name"]) + MODEL_PARAMS["causal_type"] + BIAS_PATH))

                if args.testing:
                    print('\nTESTING for Iteration', iteration, 'Intervention', intervention, "\n")
                    if DATASET == "SimBank":
                        model_params_estimator = load_data(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "MODEL_PARAMS_GENERAL_" + str(DATASET_PARAMS["intervention_name"]) + "_CE"))
                        prep_params_estimator = load_data(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "PREP_PARAMS_GENERAL_" + str(DATASET_PARAMS["intervention_name"]) + "_CE"))
                        # check first whether file exists, otherwise, grap them as above
                        if os.path.exists(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "prep_utils_" + str(DATASET_PARAMS["intervention_name"]) + str(delta) + "_LSTM" + "_CE")):
                            prep_utils_LSTM = load_data(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "prep_utils_" + str(DATASET_PARAMS["intervention_name"]) + str(delta) + "_LSTM" + "_CE"))
                            prep_utils_Vanilla_NN = load_data(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "prep_utils_" + str(DATASET_PARAMS["intervention_name"]) + str(delta) + "_Vanilla_NN" + "_CE"))
                        else:
                            # These are not the actual delta_values, is just the name of the file
                            delta_values = [0.0, 0.5, 0.75, 0.9, 0.95]
                            prep_utils_per_delta_per_model_type = load_data(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "prep_utils_per_delta_per_model_type" + str(DATASET_PARAMS["intervention_name"]) + str(delta_values) + "_CE"))
                            prep_utils_LSTM = prep_utils_per_delta_per_model_type[delta]["LSTM"]
                            prep_utils_Vanilla_NN = prep_utils_per_delta_per_model_type[delta]["Vanilla_NN"]

                        if DATASET == "SimBank":
                            generator_classes = {"ProCause": procause_generator}
                            GENERATOR_PREP_PARAMS = {"ProCause": PREP_PARAMS}
                            ESTIMATOR_PARAMS = {"prep_utils": {"LSTM": prep_utils_LSTM, "Vanilla_NN": prep_utils_Vanilla_NN},
                                                    "model_params": model_params_estimator, "prep_params": prep_params_estimator}
                            simbank_evaluator = SimBankEvaluator(DATASET_PARAMS=DATASET_PARAMS, DATASET_PARAMS_INT=DATASET_PARAMS_LIST[intervention], 
                                                                EVALUATOR_PARAMS=EVALUATOR_PARAMS, generator_classes=generator_classes, 
                                                                GENERATOR_PREP_PARAMS=GENERATOR_PREP_PARAMS, ESTIMATOR_PARAMS=ESTIMATOR_PARAMS, 
                                                                delta=delta, BIG_EVAL=args.big_eval, generator_iteration=iteration, get_dfs_policies=args.get_dfs_policies, get_dfs_policies_only=args.get_dfs_policies_only,
                                                                policy_names=args.policies,
                                                                fixed_policy_delta=args.fixed_policy_delta)
                            wssd_realcause_dict, wssd_procause_dict = simbank_evaluator.evaluate()

                            print("Wasserstein distances RealCause: ", wssd_realcause_dict)
                            print("Wasserstein distances ProCause: ", wssd_procause_dict)
                    elif 'bpic' in DATASET:
                        # SAMPLE_FOR_STAT_TESTS = False
                        SAMPLE_FOR_STAT_TESTS = True

                        # get the samples of the current iteration
                        if SAMPLE_FOR_STAT_TESTS:
                            _, _, _, cf_t_est_samples, cf_y_est_samples = procause_generator.sample(dataset="test", ret_counterfactuals=True, seed=82)
                            _, _, _, t_est_samples, y_est_samples = procause_generator.sample(dataset="test", seed=82)

                            if args.big_data:
                                save_data(cf_t_est_samples, os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "cf_t_est_samples_" + str(iteration) + "_" + str(DATASET_PARAMS["intervention_name"]) + MODEL_PARAMS["causal_type"] + BIAS_PATH + args.just_take_best_STRING))
                                save_data(cf_y_est_samples, os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "cf_y_est_samples_" + str(iteration) + "_" + str(DATASET_PARAMS["intervention_name"]) + MODEL_PARAMS["causal_type"] + BIAS_PATH + args.just_take_best_STRING))
                                save_data(t_est_samples, os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "t_est_samples_" + str(iteration) + "_" + str(DATASET_PARAMS["intervention_name"]) + MODEL_PARAMS["causal_type"] + BIAS_PATH + args.just_take_best_STRING))
                                save_data(y_est_samples, os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "y_est_samples_" + str(iteration) + "_" + str(DATASET_PARAMS["intervention_name"]) + MODEL_PARAMS["causal_type"] + BIAS_PATH + args.just_take_best_STRING))

                        EVALUATE_METRICS = True
                        # EVALUATE_METRICS = False
                        # SAMPLE_FOR_METRICS = False
                        SAMPLE_FOR_METRICS = True

                        if EVALUATE_METRICS:
                            if SAMPLE_FOR_METRICS:
                                estimated_outcome_df = procause_generator.sample_bpic(nr_samples_per_case=50, seed=82)
                                if args.big_data:
                                    save_data(estimated_outcome_df, os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "estimated_outcome_df_" + str(iteration) + "_" + str(DATASET_PARAMS["intervention_name"]) + MODEL_PARAMS["causal_type"] + BIAS_PATH + args.just_take_best_STRING))
                            else:
                                estimated_outcome_df = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "estimated_outcome_df_" + str(iteration) + "_" + str(DATASET_PARAMS["intervention_name"]) + MODEL_PARAMS["causal_type"] + BIAS_PATH + args.just_take_best_STRING))
                            metrics = procause_generator.calculate_metrics_bpic(estimated_outcome_df_list=estimated_outcome_df)
                            if args.big_data:
                                save_data(metrics, os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "metrics_" + str(iteration) + "_" + str(DATASET_PARAMS["intervention_name"]) + MODEL_PARAMS["causal_type"] + BIAS_PATH + args.just_take_best_STRING))

if args.testing and 'bpic' in DATASET:
    # EVALUATE ENSEMBLE METRICS
    for iteration in range(args.num_iterations):
        estimated_outcome_df_list = []
        for learner in args.learners:
            estimated_outcome_df = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "estimated_outcome_df_" + str(iteration) + "_" + str(DATASET_PARAMS["intervention_name"]) + learner + BIAS_PATH + args.just_take_best_STRING))
            estimated_outcome_df_list.append(estimated_outcome_df)
        metrics_ensemble, merged_df = procause_generator.calculate_metrics_bpic(estimated_outcome_df_list=estimated_outcome_df_list, ensemble=True)
        print('Metrics ensemble for iteration', iteration, metrics_ensemble, "\n")
        if args.big_data:
            save_data(metrics_ensemble, os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "metrics_ensemble_" + str(iteration) + "_" + str(DATASET_PARAMS["intervention_name"]) + BIAS_PATH + args.just_take_best_STRING))
            save_data(merged_df, os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "merged_df_ensemble_" + str(iteration) + "_" + str(DATASET_PARAMS["intervention_name"]) + BIAS_PATH + args.just_take_best_STRING))
    # EVALUATE STATISTICAL TESTS
    # get all the samples for each learner over all iterations and pool them in a tensor, then do the statistical tests
    y_est_samples_ensemble_all = []
    for learner in args.learners:
        print("Evaluating learner", learner, "for dataset", DATASET, "\n")
        y_est_samples_all = []
        t_est_samples_all = []
        for iteration in range(args.num_iterations):
            y_est_samples = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "y_est_samples_" + str(iteration) + "_" + str(DATASET_PARAMS["intervention_name"]) + learner + BIAS_PATH + args.just_take_best_STRING))
            y_est_samples_all.append(y_est_samples)
            t_est_samples = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "t_est_samples_" + str(iteration) + "_" + str(DATASET_PARAMS["intervention_name"]) + "TarNet" + BIAS_PATH + args.just_take_best_STRING))
            t_est_samples_all.append(t_est_samples)
            print("y_est_samples.shape", y_est_samples.shape)
            print("t_est_samples.shape", t_est_samples.shape)
        print("y_est_samples_all.shape", len(y_est_samples_all))
        print("t_est_samples_all.shape", len(t_est_samples_all))
        y_est_samples_all = torch.stack(y_est_samples_all, dim=0)
        y_est_samples_ensemble_all.append(y_est_samples_all)
        print("y_est_samples_all.shape after stacking", y_est_samples_all.shape)
        # also make sure to stack t, but it should still be an array
        t_est_samples_all = np.array(t_est_samples_all)
        if args.big_data:
            save_data(t_est_samples_all, os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "t_est_samples_all_" + str(DATASET_PARAMS["intervention_name"]) + learner + BIAS_PATH + args.just_take_best_STRING))
            save_data(y_est_samples_all, os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "y_est_samples_all_" + str(DATASET_PARAMS["intervention_name"]) + learner + BIAS_PATH + args.just_take_best_STRING))

        uni_metrics_test, _, _ = procause_generator.evaluate_statistical(y_model=y_est_samples_all, t_model=t_est_samples_all, only_univariate=True)
        if args.big_data:
            save_data(uni_metrics_test, os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "uni_metrics_test_all_" + str(DATASET_PARAMS["intervention_name"]) + learner + BIAS_PATH + args.just_take_best_STRING))
        print('\n')

    print("Evaluating ensemble for dataset", DATASET, "\n")
    y_est_ensemble = torch.stack(y_est_samples_ensemble_all, dim=0).mode(dim=0).values
    print("y_est_ensemble.shape", y_est_ensemble.shape)

    uni_metrics_test, multi_metrics_test_no_x, multi_metrics_test = procause_generator.evaluate_statistical(y_model=y_est_ensemble, t_model=t_est_samples_all)
    if args.big_data:
        save_data(uni_metrics_test, os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "uni_metrics_test_ensemble_all_" + str(DATASET_PARAMS["intervention_name"]) + BIAS_PATH + args.just_take_best_STRING))
        save_data(multi_metrics_test_no_x, os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "multi_metrics_test_ensemble_all_no_x_" + str(DATASET_PARAMS["intervention_name"]) + BIAS_PATH + args.just_take_best_STRING))
        save_data(multi_metrics_test, os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "multi_metrics_test_ensemble_all_" + str(DATASET_PARAMS["intervention_name"]) + BIAS_PATH + args.just_take_best_STRING))
    print('\n')