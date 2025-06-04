DATA_FOLDER = "data"
RESULTS_FOLDER = "res"
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from config.config import path
sys.path.append(path)
# sys.path.append(path + "\\SimBank")
# sys.path.append(path + "\\src\\methods\\BOZORGI")

from src.utils.tools import save_data, load_data
from copy import deepcopy
import torch
from src.causal_estimators.main_causal_estimators import CausalEstimator
from src.utils.preprocessor.main_prep import ProcessPreprocessor
from SimBank.confounding_level import set_delta
import argparse
import json

parser = argparse.ArgumentParser(description='CE')
parser.add_argument('--config', type=str, help='Path to config file')
args, unknown = parser.parse_known_args()
config_args = {}
if args.config:
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_args = json.load(f)

parser.add_argument('--intervention_name', nargs='+', type=str, default=config_args.get('intervention_name', ["set_ir_3_levels"]), help='Intervention name')
parser.add_argument('--already_preprocessed', type=lambda x: x.lower() == 'true', default=config_args.get('already_preprocessed', False), help='Already preprocessed (True or False)')

args = parser.parse_args()
print(args)

# BIG_DATA = False
BIG_DATA = True
# ALREADY_PREPROCESSED = False
ALREADY_PREPROCESSED = args.already_preprocessed
NUM_ITERATIONS = 5
ITERATIONS_TO_SKIP = []

# Say which causal estimators to run
# ESTIMATORS = ["TarNet", "S-Learner", "T-Learner"]
MODEL_TYPES = ["LSTM", "Vanilla_NN"]
ESTIMATORS = ["TarNet", "S-Learner", "T-Learner"]
# MODEL_TYPES = ["Vanilla_NN"]
# delta_values = [0.0, 0.5, 0.75, 0.9, 0.95]
# delta_values = [0.8, 0.85, 0.91, 0.92, 0.93, 0.94, 0.96, 0.97, 0.98, 0.99, 0.999]
delta_values = [0.0, 0.5, 0.75, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999]
# NR_THRESHOLDS = 10
NR_THRESHOLDS = 20
# delta_values = [0.95]

# Say which dataset (and whether threshold tuning is needed)
DATASET = "SimBank"
DATASET_PARAMS = {}
# Test prop --> validation set for threshold tuning
PREP_PARAMS = {"test_prop": 0.1, "val_prop": 0.2, "standardize": True, "one_hot_encode": True, "impute": True, "seed": 0} 
MODEL_PARAMS = {"early_stop": True, "eval_every": 500, "num_epochs": 100000, "masked": True, "disable_progress_bar": True, "print_every_iters": 2000}

if DATASET == "SimBank":
    DATASET_PARAMS["intervention_name"] = args.intervention_name
    # DATASET_PARAMS["intervention_name"] = ["time_contact_HQ"]
    dataset_params_simbank = load_data(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "loan_log_" + str(DATASET_PARAMS["intervention_name"]) + "_" + str(100000) + "_dataset_params"))
    DATASET_PARAMS.update(dataset_params_simbank)
    DATASET_PARAMS["train_size"] = 2500
    DATASET_PARAMS["data_path_normal"] = "loan_log_" + str(DATASET_PARAMS["intervention_name"]) + "_" + str(100000) + "_train_normal"
    # LOAD DATA
    data_normal = load_data(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, DATASET_PARAMS["data_path_normal"]))
    DATASET_PARAMS["data_path_RCT"] = "loan_log_" + str(DATASET_PARAMS["intervention_name"]) + "_" + str(100000) + "_train_RCT"
    data_RCT = load_data(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, DATASET_PARAMS["data_path_RCT"]))
    
    DATASET_PARAMS["case_nr_column"] = "case_nr"
    DATASET_PARAMS["order_column"] = "timestamp"
    DATASET_PARAMS["event_cols"].remove("cum_cost")
    DATASET_PARAMS["scale_cols"].remove("cum_cost")
    DATASET_PARAMS["nr_of_interventions"] = len(DATASET_PARAMS["intervention_name"])
    if DATASET_PARAMS["intervention_name"][0] == "set_ir_3_levels":
        DATASET_PARAMS["action_width"] = [3]
        DATASET_PARAMS["action_depth"] = [1]
    elif DATASET_PARAMS["intervention_name"][0] == "time_contact_HQ":
        DATASET_PARAMS["action_width"] = [2]
        DATASET_PARAMS["action_depth"] = [4]

    PREP_PARAMS["filter_useless_cols"] = True
    PREP_PARAMS["calc_atoms"] = False
    PREP_PARAMS["last_state_cols"] = ["elapsed_time", "cum_cost"]
    PREP_PARAMS["last_state_cols"].remove("cum_cost")

    MODEL_PARAMS["lr"] = 0.0001
    MODEL_PARAMS["batch_size"] = 1024
    MODEL_PARAMS["dim_y"] = 1
    MODEL_PARAMS["dim_t"] = 1 if DATASET_PARAMS["action_width"][0] == 2 else DATASET_PARAMS["action_width"][0]
    MODEL_PARAMS["patience"] = 7500
    MODEL_PARAMS["grad_norm"] = 1.0
    
    MODEL_PARAMS["n_lstm"] = 1
    MODEL_PARAMS["n_dense_in_lstm"] = 1
    MODEL_PARAMS["dim_hidden_lstm"] = 25

    MODEL_PARAMS["n_dense"] = 2
    MODEL_PARAMS["dim_hidden_dense"] = 64

if not BIG_DATA:
    MODEL_PARAMS["num_epochs"] = 10
    MODEL_PARAMS["patience"] = 10
    MODEL_PARAMS["lr"] = 0.001
    MODEL_PARAMS["print_every_iters"] = 10
    MODEL_PARAMS["eval_every"] = 10
# save MODEL_PARAMS and PREP_PARAMS
save_data(MODEL_PARAMS, os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "MODEL_PARAMS_GENERAL_" + str(DATASET_PARAMS["intervention_name"]) + "_CE"))
save_data(PREP_PARAMS, os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "PREP_PARAMS_GENERAL_" + str(DATASET_PARAMS["intervention_name"]) + "_CE"))
# MODEL_PARAMS["num_epochs"] = 10
# MODEL_PARAMS["patience"] = 10
# MODEL_PARAMS["lr"] = 0.001
# MODEL_PARAMS["print_every_iters"] = 10
# MODEL_PARAMS["eval_every"] = 10

data_train_prep_per_delta_per_model_type = {}
data_val_prep_per_delta_per_model_type = {}
data_val_th_prep_per_delta_per_model_type = {}
prep_utils_per_delta_per_model_type = {}

if not ALREADY_PREPROCESSED:
    for delta in delta_values:
        # cut the data before the setting of the delta (IMPORTANT FOR VARIATION IN RESULTS)
        case_nrs_normal = data_normal[DATASET_PARAMS["case_nr_column"]].unique()
        case_nrs_normal = case_nrs_normal[:DATASET_PARAMS["train_size"]]
        data_normal = data_normal[data_normal[DATASET_PARAMS["case_nr_column"]].isin(case_nrs_normal)]

        case_nrs_RCT = data_RCT[DATASET_PARAMS["case_nr_column"]].unique()
        case_nrs_RCT = case_nrs_RCT[:DATASET_PARAMS["train_size"]]
        data_RCT = data_RCT[data_RCT[DATASET_PARAMS["case_nr_column"]].isin(case_nrs_RCT)]

        data = set_delta(data=data_normal, data_RCT=data_RCT, delta=delta)

        if not BIG_DATA:
            data = data.head(10000)
        # data = data.head(10000)

        data_train_prep_per_delta_per_model_type[delta] = {}
        data_val_prep_per_delta_per_model_type[delta] = {}
        data_val_th_prep_per_delta_per_model_type[delta] = {}
        prep_utils_per_delta_per_model_type[delta] = {}
        for model_type in MODEL_TYPES:
            # PREPROCESS DATA (for each intervention)
            data_train_prep_list = []
            data_val_prep_list = []
            data_val_th_prep_list = []
            prep_utils_list = []

            DATASET_PARAMS_LIST = []
            for int in range(DATASET_PARAMS["nr_of_interventions"]):
                params = deepcopy(DATASET_PARAMS)
                for key, value in params["intervention_info"].items():
                    if isinstance(value, list):
                        params["intervention_info"][key] = value[int]
                DATASET_PARAMS_LIST.append(params)

            if model_type in ["LSTM", "CNN", "LSTM-VAE"]:
                PREP_PARAMS["encoding"] = "sequential"
            else:
                PREP_PARAMS["encoding"] = "aggregated"

            print("\nPREPROCESSING: encoding", PREP_PARAMS["encoding"], "model_type", model_type, "delta", delta)
            for int in range(DATASET_PARAMS["nr_of_interventions"]):
                preprocessor = ProcessPreprocessor(DATASET = DATASET, raw_data = data, PREP_PARAMS = PREP_PARAMS, DATASET_PARAMS = DATASET_PARAMS_LIST[int])
                data_train_prep, data_val_prep, data_val_th_prep, prep_utils = preprocessor.preprocess()
                data_train_prep_list.append(data_train_prep)
                data_val_prep_list.append(data_val_prep)
                data_val_th_prep_list.append(data_val_th_prep)
                prep_utils_list.append(prep_utils)
            
            data_train_prep_per_delta_per_model_type[delta][model_type] = data_train_prep_list
            data_val_prep_per_delta_per_model_type[delta][model_type] = data_val_prep_list
            data_val_th_prep_per_delta_per_model_type[delta][model_type] = data_val_th_prep_list
            prep_utils_per_delta_per_model_type[delta][model_type] = prep_utils_list

    if BIG_DATA:
        # save per delta and model type
        for delta in delta_values:
            for model_type in MODEL_TYPES:
                data_train_prep = data_train_prep_per_delta_per_model_type[delta][model_type]
                data_val_prep = data_val_prep_per_delta_per_model_type[delta][model_type]
                data_val_th_prep = data_val_th_prep_per_delta_per_model_type[delta][model_type]
                prep_utils = prep_utils_per_delta_per_model_type[delta][model_type]

                save_data(data_train_prep, os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "data_train_prep_" + str(DATASET_PARAMS["intervention_name"]) + str(delta) + "_" + model_type + "_CE"))
                save_data(data_val_prep, os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "data_val_prep_" + str(DATASET_PARAMS["intervention_name"]) + str(delta) + "_" + model_type + "_CE"))
                save_data(data_val_th_prep, os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "data_val_th_prep_" + str(DATASET_PARAMS["intervention_name"]) + str(delta) + "_" + model_type + "_CE"))
                save_data(prep_utils, os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "prep_utils_" + str(DATASET_PARAMS["intervention_name"]) + str(delta) + "_" + model_type + "_CE"))

        # save_data(data_train_prep_per_delta_per_model_type, os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "data_train_prep_per_delta_per_model_type_" + str(DATASET_PARAMS["intervention_name"]) + str(delta_values) + "_CE"))
        # save_data(data_val_prep_per_delta_per_model_type, os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "data_val_prep_per_delta_per_model_type" + str(DATASET_PARAMS["intervention_name"]) + str(delta_values) + "_CE"))
        # save_data(data_val_th_prep_per_delta_per_model_type, os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "data_val_th_prep_per_delta_per_model_type" + str(DATASET_PARAMS["intervention_name"]) + str(delta_values) + "_CE"))
        # save_data(prep_utils_per_delta_per_model_type, os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "prep_utils_per_delta_per_model_type" + str(DATASET_PARAMS["intervention_name"]) + str(delta_values) + "_CE"))

else:
    # load per delta and model type
    for delta in delta_values:
        data_train_prep_per_delta_per_model_type[delta] = {}
        data_val_prep_per_delta_per_model_type[delta] = {}
        data_val_th_prep_per_delta_per_model_type[delta] = {}
        prep_utils_per_delta_per_model_type[delta] = {}
        for model_type in MODEL_TYPES:
            data_train_prep_per_delta_per_model_type[delta][model_type] = load_data(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "data_train_prep_" + str(DATASET_PARAMS["intervention_name"]) + str(delta) + "_" + model_type + "_CE"))
            data_val_prep_per_delta_per_model_type[delta][model_type] = load_data(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "data_val_prep_" + str(DATASET_PARAMS["intervention_name"]) + str(delta) + "_" + model_type + "_CE"))
            data_val_th_prep_per_delta_per_model_type[delta][model_type] = load_data(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "data_val_th_prep_" + str(DATASET_PARAMS["intervention_name"]) + str(delta) + "_" + model_type + "_CE"))
            prep_utils_per_delta_per_model_type[delta][model_type] = load_data(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "prep_utils_" + str(DATASET_PARAMS["intervention_name"]) + str(delta) + "_" + model_type + "_CE"))

    # data_train_prep_per_delta_per_model_type = load_data(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "data_train_prep_per_delta_per_model_type_" + str(DATASET_PARAMS["intervention_name"]) + str(delta_values) + "_CE"))
    # data_val_prep_per_delta_per_model_type = load_data(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "data_val_prep_per_delta_per_model_type" + str(DATASET_PARAMS["intervention_name"]) + str(delta_values) + "_CE"))
    # data_val_th_prep_per_delta_per_model_type = load_data(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "data_val_th_prep_per_delta_per_model_type" + str(DATASET_PARAMS["intervention_name"]) + str(delta_values) + "_CE"))
    # prep_utils_per_delta_per_model_type = load_data(os.path.join(os.getcwd(), DATA_FOLDER, DATASET, "prep_utils_per_delta_per_model_type" + str(DATASET_PARAMS["intervention_name"]) + str(delta_values) + "_CE"))

for int in range(DATASET_PARAMS["nr_of_interventions"]):
    for delta in delta_values:
        for model_type in MODEL_TYPES:
            for estimator in ESTIMATORS:
                for iteration in range(NUM_ITERATIONS):
                    if iteration in ITERATIONS_TO_SKIP:
                        continue
                    data_train_prep = data_train_prep_per_delta_per_model_type[delta][model_type][int]
                    data_val_prep = data_val_prep_per_delta_per_model_type[delta][model_type][int]
                    data_val_th_prep = data_val_th_prep_per_delta_per_model_type[delta][model_type][int]
                    prep_utils = prep_utils_per_delta_per_model_type[delta][model_type][int]

                    MODEL_PARAMS["model_type"] = model_type
                    MODEL_PARAMS["causal_type"] = estimator
                    MODEL_PARAMS["seed"] = 99 + iteration*5
                    MODEL_PARAMS["savepath"] = os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "cache_causal_estimator_" + str(DATASET_PARAMS["intervention_name"]) + str(delta) + "_" + model_type + "_" + estimator + "_" + str(iteration) + "_CE")

                    # Train
                    print("\nTRAINING: delta", delta, "model_type", model_type, "estimator", estimator, "iteration", iteration)
                    causal_estimator = CausalEstimator(data_train=data_train_prep, data_val=data_val_prep, data_val_th=data_val_th_prep, prep_utils=prep_utils, MODEL_PARAMS=MODEL_PARAMS)
                    causal_estimator.train()

                    # Save
                    if BIG_DATA:
                        torch.save([net.state_dict() for net in causal_estimator.networks], os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "causal_estimator_networks_" + str(DATASET_PARAMS["intervention_name"]) + str(delta) + "_" + model_type + "_" + estimator + "_" + str(iteration) + "_CE"))

                    # Tune threshold if necessary
                    opt_th = 0
                    best_est_perf = 0
                    if DATASET_PARAMS["action_depth"][0] > 1:
                        print("\nTUNING THRESHOLD: delta", delta, "model_type", model_type, "estimator", estimator, "iteration", iteration)
                        opt_th, best_est_perf = causal_estimator.tune_threshold(nr_thresholds=NR_THRESHOLDS)

                        if BIG_DATA:
                            save_data(opt_th, os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "opt_th_" + str(DATASET_PARAMS["intervention_name"]) + str(delta) + "_" + model_type + "_" + estimator + "_" + str(iteration) + "_CE"))
                            save_data(best_est_perf, os.path.join(os.getcwd(), RESULTS_FOLDER, DATASET, "best_est_perf_" + str(DATASET_PARAMS["intervention_name"]) + str(delta) + "_" + model_type + "_" + estimator + "_" + str(iteration) + "_CE"))