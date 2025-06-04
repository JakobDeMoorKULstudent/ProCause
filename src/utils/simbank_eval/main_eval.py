from SimBank.extra_flow_conditions import ExtraFlowConditioner
from src.utils.simbank_eval.policies_to_eval import get_bank_best_action, get_random_best_timing, get_random_best_action, get_causal_estimator_action
import src.utils.preprocessor.simbank_prep as simbank_prep
from SimBank import simulation
import pandas as pd
import random
import torch
from scipy import stats
from src.causal_estimators.main_causal_estimators import CausalEstimator
from src.utils.tools import load_data, save_data
import os
DATA_FOLDER = "data"
RESULTS_FOLDER = "res"

class SimBankEvaluator():
    def __init__(self, DATASET_PARAMS, DATASET_PARAMS_INT, EVALUATOR_PARAMS, generator_classes, GENERATOR_PREP_PARAMS, ESTIMATOR_PARAMS, print_cases=False, delta=0.95, BIG_EVAL=True, generator_iteration=0, get_dfs_policies=True, get_dfs_policies_only=False, policy_names=["bank", "random"], fixed_policy_delta=None):
        self.DATASET_PARAMS = DATASET_PARAMS
        self.DATASET_PARAMS_INT = DATASET_PARAMS_INT
        self.EVALUATOR_PARAMS = EVALUATOR_PARAMS
        self.generator_classes = generator_classes
        self.GENERATOR_PREP_PARAMS = GENERATOR_PREP_PARAMS
        self.ESTIMATOR_PARAMS = ESTIMATOR_PARAMS
        self.print_cases = print_cases
        self.BIG_EVAL = BIG_EVAL
        self.generator_iteration = generator_iteration
        self.get_dfs_policies = get_dfs_policies
        self.get_dfs_policies_only = get_dfs_policies_only
        self.policy_names = policy_names

        self.delta_evaluator = delta
        self.delta_policy = delta
        self.fixed_policy_delta_to_add = ""
        if fixed_policy_delta is not None:
            self.delta_policy = fixed_policy_delta
            self.fixed_policy_delta_to_add = "_fixed_" + str(fixed_policy_delta)

        self.evaluation_type_suffix = ""
        if EVALUATOR_PARAMS["nr_cases"] != 1000 and EVALUATOR_PARAMS["nr_samples_per_case"] != 50:
            self.evaluation_type_suffix = "_eval_" + str(EVALUATOR_PARAMS["nr_cases"]) + "_" + str(EVALUATOR_PARAMS["nr_samples_per_case"])

        self.case_preps = {}
        if "RealCause" in self.GENERATOR_PREP_PARAMS.keys():
            self.case_preps["RealCause"] = simbank_prep.SimBankPreprocessor(DATASET_PARAMS=self.DATASET_PARAMS_INT, PREP_PARAMS=GENERATOR_PREP_PARAMS["RealCause"])
        if "ProCause" in self.GENERATOR_PREP_PARAMS.keys():
            self.case_preps["ProCause"] = simbank_prep.SimBankPreprocessor(DATASET_PARAMS=self.DATASET_PARAMS_INT, PREP_PARAMS=GENERATOR_PREP_PARAMS["ProCause"])
        
        self.case_preps["estimator"] = simbank_prep.SimBankPreprocessor(DATASET_PARAMS=self.DATASET_PARAMS_INT, PREP_PARAMS=self.ESTIMATOR_PARAMS["prep_params"])

    def evaluate(self):
        self.suffix = "RealCause"
        # if self.BIG_EVAL:
        #     if "RealCause" in self.generator_classes.keys():
        #         self.suffix = "RealCause"
        #     if "ProCause" in self.generator_classes.keys():
        #         self.suffix = "ProCause" + str(self.generator_classes["ProCause"].MODEL_PARAMS["causal_type"])

        online_performances, online_outcome_dfs, online_full_dfs, online_actions_dfs, offline_full_dfs, offline_actions_dfs = self.get_true_outcomes_and_offline_dfs()
        if self.BIG_EVAL:

            # NOTE: to change again

            save_data(online_performances, os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "online_performances_" + str(self.DATASET_PARAMS["intervention_name"]) + str(self.delta_policy) + "_generator_iteration" + str(0) + self.suffix + str(self.policy_names) + str(self.evaluation_type_suffix)))
            save_data(online_outcome_dfs, os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "online_outcome_dfs_" + str(self.DATASET_PARAMS["intervention_name"]) + str(self.delta_policy) + "_generator_iteration" + str(0) + self.suffix + str(self.policy_names) + str(self.evaluation_type_suffix)))
            save_data(online_full_dfs, os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "online_full_dfs_" + str(self.DATASET_PARAMS["intervention_name"]) + str(self.delta_policy) + "_generator_iteration" + str(0) + self.suffix + str(self.policy_names) + str(self.evaluation_type_suffix)))
            save_data(online_actions_dfs, os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "online_actions_dfs_" + str(self.DATASET_PARAMS["intervention_name"]) + str(self.delta_policy) + "_generator_iteration" + str(0) + self.suffix + str(self.policy_names) + str(self.evaluation_type_suffix)))
            
            save_data(offline_full_dfs, os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "offline_full_dfs_" + str(self.DATASET_PARAMS["intervention_name"]) + str(self.delta_policy) + "_generator_iteration" + str(0) + self.suffix + str(self.policy_names) + str(self.evaluation_type_suffix)))
            save_data(offline_actions_dfs, os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "offline_actions_dfs_" + str(self.DATASET_PARAMS["intervention_name"]) + str(self.delta_policy) + "_generator_iteration" + str(0) + self.suffix + str(self.policy_names) + str(self.evaluation_type_suffix)))
        
        # NOTE: to change 
        


        # prep_agg_offline_full_dict, prep_seq_offline_full_dict = self.prep_offline_full_dfs(offline_full_dfs)

        prep_agg_offline_full_dict, prep_seq_offline_full_dict = self.prep_offline_full_dfs(online_full_dfs)
        if self.BIG_EVAL:
            if "RealCause" in self.generator_classes.keys():
                save_data(prep_agg_offline_full_dict, os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "prep_agg_offline_full_dict_" + str(self.DATASET_PARAMS["intervention_name"]) + str(self.delta_policy) + "_generator_iteration" + str(0) + "RealCause" + str(self.policy_names) + str(self.evaluation_type_suffix)))
            if "ProCause" in self.generator_classes.keys():
                save_data(prep_seq_offline_full_dict, os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "prep_seq_offline_full_dict_" + str(self.DATASET_PARAMS["intervention_name"]) + str(self.delta_policy) + "_generator_iteration" + str(0) + "ProCause" + str(self.policy_names) + str(self.evaluation_type_suffix)))

        if self.get_dfs_policies_only:
            return {}, {}
        
        # NOTE: to change again

        # estimated_outcome_dfs_realcause, estimated_outcome_dfs_procause = self.get_estimated_outcomes(prep_agg_offline_full_dict, prep_seq_offline_full_dict, offline_actions_dfs)

        estimated_outcome_dfs_realcause, estimated_outcome_dfs_procause = self.get_estimated_outcomes(prep_agg_offline_full_dict, prep_seq_offline_full_dict, online_actions_dfs)
        if self.BIG_EVAL:
            if "RealCause" in self.generator_classes.keys():
                save_data(estimated_outcome_dfs_realcause, os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "estimated_outcome_dfs_realcause_" + str(self.DATASET_PARAMS["intervention_name"]) + str(self.delta_evaluator) + "_generator_iteration" + str(self.generator_iteration) + "RealCause" + str(self.generator_classes["RealCause"].MODEL_PARAMS["causal_type"] + str(self.policy_names) + str(self.evaluation_type_suffix) + self.fixed_policy_delta_to_add)))
            if "ProCause" in self.generator_classes.keys():
                save_data(estimated_outcome_dfs_procause, os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "estimated_outcome_dfs_procause_" + str(self.DATASET_PARAMS["intervention_name"]) + str(self.delta_evaluator) + "_generator_iteration" + str(self.generator_iteration) + "ProCause" + str(self.generator_classes["ProCause"].MODEL_PARAMS["causal_type"] + str(self.policy_names) + str(self.evaluation_type_suffix) + self.fixed_policy_delta_to_add)))

        wssd_dict_realcause, wssd_dict_procause = self.get_wssds(online_outcome_dfs, estimated_outcome_dfs_realcause, estimated_outcome_dfs_procause)
        if self.BIG_EVAL:
            if "RealCause" in self.generator_classes.keys():
                save_data(wssd_dict_realcause, os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "wssd_dict_realcause_" + str(self.DATASET_PARAMS["intervention_name"]) + str(self.delta_evaluator) + "_generator_iteration" + str(self.generator_iteration) + "RealCause" + str(self.generator_classes["RealCause"].MODEL_PARAMS["causal_type"] + str(self.policy_names) + str(self.evaluation_type_suffix) + self.fixed_policy_delta_to_add)))
            if "ProCause" in self.generator_classes.keys():
                save_data(wssd_dict_procause, os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "wssd_dict_procause_" + str(self.DATASET_PARAMS["intervention_name"]) + str(self.delta_evaluator) + "_generator_iteration" + str(self.generator_iteration) + "ProCause" + str(self.generator_classes["ProCause"].MODEL_PARAMS["causal_type"] + str(self.policy_names) + str(self.evaluation_type_suffix) + self.fixed_policy_delta_to_add)))
        # wssd_dict_realcause = {}
        # wssd_dict_procause = {}
        return wssd_dict_realcause, wssd_dict_procause
    
    def get_true_outcomes_and_offline_dfs(self):
        self.online_performances = {}
        self.online_outcome_dfs = {}
        self.online_full_dfs = {}
        self.online_actions_dfs = {}
        
        self.offline_full_dfs = {}
        self.offline_actions_dfs = {}

        # NOTE: to change again

        # self.bank_test_df = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "bank_test_df_" + str(self.DATASET_PARAMS["intervention_name"]) + "0.95" + "_generator_iteration" + str(0) + self.suffix + str(self.evaluation_type_suffix)))
        # self.bank_performance = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "online_performances_" + str(self.DATASET_PARAMS["intervention_name"]) + "0.95" + "_generator_iteration" + str(0) + self.suffix + "bank" + str(self.evaluation_type_suffix)))
        # self.bank_outcome_df = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "online_outcome_dfs_" + str(self.DATASET_PARAMS["intervention_name"]) + "0.95" + "_generator_iteration" + str(0) + self.suffix + "bank" + str(self.evaluation_type_suffix)))

        # _, _, _, self.bank_online_full_df, self.bank_online_action_df = self.policy_online(self.EVALUATOR_PARAMS["nr_cases"], {"name": "bank"})
        
        # check if the file already exists, if not save it, otherwise load it
        if self.BIG_EVAL and os.path.exists(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "bank_test_df_" + str(self.DATASET_PARAMS["intervention_name"]) + "0.95" + "_generator_iteration" + str(0) + self.suffix + str(self.evaluation_type_suffix))) and not self.get_dfs_policies:
                self.bank_test_df = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "bank_test_df_" + str(self.DATASET_PARAMS["intervention_name"]) + "0.95" + "_generator_iteration" + str(0) + self.suffix + str(self.evaluation_type_suffix)))
                self.bank_performance = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "online_performances_" + str(self.DATASET_PARAMS["intervention_name"]) + "0.95" + "_generator_iteration" + str(0) + self.suffix + "bank" + str(self.evaluation_type_suffix)))
                self.bank_outcome_df = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "online_outcome_dfs_" + str(self.DATASET_PARAMS["intervention_name"]) + "0.95" + "_generator_iteration" + str(0) + self.suffix + "bank" + str(self.evaluation_type_suffix)))
                self.bank_online_full_df = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "online_full_dfs_" + str(self.DATASET_PARAMS["intervention_name"]) + "0.95" + "_generator_iteration" + str(0) + self.suffix + "bank" + str(self.evaluation_type_suffix)))
                self.bank_online_action_df = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "online_actions_dfs_" + str(self.DATASET_PARAMS["intervention_name"]) + "0.95" + "_generator_iteration" + str(0) + self.suffix + "bank" + str(self.evaluation_type_suffix)))
        else:
            # Get Bank policy performance, which is needed for the estimated performance
            self.bank_performance, self.bank_outcome_df, self.bank_test_df, self.bank_online_full_df, self.bank_online_action_df = self.policy_online(self.EVALUATOR_PARAMS["nr_cases"], {"name": "bank"})
        
        self.online_performances["bank"] = self.bank_performance
        self.online_outcome_dfs["bank"] = self.bank_outcome_df
        self.online_full_dfs["bank"] = [self.bank_online_full_df]
        self.online_actions_dfs["bank"] = self.bank_online_action_df

        self.offline_full_dfs["bank"] = [self.bank_test_df]

        # save the bank

        # NOTE: to change again

        if self.BIG_EVAL:
            save_data(self.online_performances["bank"], os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "online_performances_" + str(self.DATASET_PARAMS["intervention_name"]) + "0.95" + "_generator_iteration" + str(0) + self.suffix + "bank" + str(self.evaluation_type_suffix)))
            save_data(self.online_outcome_dfs["bank"], os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "online_outcome_dfs_" + str(self.DATASET_PARAMS["intervention_name"]) + "0.95" + "_generator_iteration" + str(0) + self.suffix + "bank" + str(self.evaluation_type_suffix)))
            save_data(self.bank_online_full_df, os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "online_full_dfs_" + str(self.DATASET_PARAMS["intervention_name"]) + "0.95" + "_generator_iteration" + str(0) + self.suffix + "bank" + str(self.evaluation_type_suffix)))
            save_data(self.bank_online_action_df, os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "online_actions_dfs_" + str(self.DATASET_PARAMS["intervention_name"]) + "0.95" + "_generator_iteration" + str(0) + self.suffix + "bank" + str(self.evaluation_type_suffix)))

            save_data(self.bank_test_df, os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "bank_test_df_" + str(self.DATASET_PARAMS["intervention_name"]) + "0.95" + "_generator_iteration" + str(0) + self.suffix + str(self.evaluation_type_suffix)))

        self.policy_estimators = {}
        # First get the true performance of each policy, then get the estimated performance (without bank policy)
        for policy in self.EVALUATOR_PARAMS["policies"]:
            print("Policy: ", policy["name"])
            if policy["name"] == "bank":
                continue
            delta = self.delta_policy
            # to get the offline dfs, the random policy is not influenced by the delta (confounding)
            if policy["name"] == "random":
                delta = 0.95
            # check if the file already exists (which is saved later on)
            if self.BIG_EVAL:
                if os.path.exists(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "offline_actions_dfs_" + str(self.DATASET_PARAMS["intervention_name"]) + str(delta) + "_generator_iteration" + str(0) + self.suffix + str(policy["name"]) + str(self.evaluation_type_suffix))) and not self.get_dfs_policies:
                    
                    self.online_performances[policy["name"]] = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "online_performances_" + str(self.DATASET_PARAMS["intervention_name"]) + str(delta) + "_generator_iteration" + str(0) + self.suffix + str(policy["name"]) + str(self.evaluation_type_suffix)))
                    self.online_outcome_dfs[policy["name"]] = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "online_outcome_dfs_" + str(self.DATASET_PARAMS["intervention_name"]) + str(delta) + "_generator_iteration" + str(0) + self.suffix + str(policy["name"]) + str(self.evaluation_type_suffix)))
                    self.online_full_dfs[policy["name"]] = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "online_full_dfs_" + str(self.DATASET_PARAMS["intervention_name"]) + str(delta) + "_generator_iteration" + str(0) + self.suffix + str(policy["name"]) + str(self.evaluation_type_suffix)))
                    self.online_actions_dfs[policy["name"]] = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "online_actions_dfs_" + str(self.DATASET_PARAMS["intervention_name"]) + str(delta) + "_generator_iteration" + str(0) + self.suffix + str(policy["name"]) + str(self.evaluation_type_suffix)))

                    self.offline_full_dfs[policy["name"]] = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "offline_full_dfs_" + str(self.DATASET_PARAMS["intervention_name"]) + str(delta) + "_generator_iteration" + str(0) + self.suffix + str(policy["name"]) + str(self.evaluation_type_suffix)))
                    self.offline_actions_dfs[policy["name"]] = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "offline_actions_dfs_" + str(self.DATASET_PARAMS["intervention_name"]) + str(delta) + "_generator_iteration" + str(0) + self.suffix + str(policy["name"]) + str(self.evaluation_type_suffix)))
                    continue

            self.online_performances[policy["name"]] = []
            self.online_outcome_dfs[policy["name"]] = []
            self.online_full_dfs[policy["name"]] = []
            self.online_actions_dfs[policy["name"]] = []
            
            self.offline_full_dfs[policy["name"]] = []
            self.offline_actions_dfs[policy["name"]] = []

            if policy["name"] not in ["random", "bank"]:
                self.ESTIMATOR_PARAMS["model_params"]["model_type"] = policy["model_type"]
                self.ESTIMATOR_PARAMS["model_params"]["causal_type"] = policy["causal_type"]
                self.ESTIMATOR_PARAMS["model_params"]["seed"] = 99
                causal_estimator = CausalEstimator(data_train=[], data_val=[], data_val_th=[], prep_utils=self.ESTIMATOR_PARAMS["prep_utils"][policy["model_type"]][0], MODEL_PARAMS=self.ESTIMATOR_PARAMS["model_params"], inference=True)
                self.policy_estimators[policy["name"]] = causal_estimator
           
            # TRUE PERFORMANCE
            for iteration in range(self.EVALUATOR_PARAMS["num_iterations"]):
                if policy["name"] not in ["bank", "random"]:
                    self.opt_th = 0
                    if self.DATASET_PARAMS["action_depth"][0] > 1:
                        self.opt_th = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "opt_th_" + str(self.DATASET_PARAMS["intervention_name"]) + str(delta) + "_" + policy["model_type"] + "_" + policy["causal_type"] + "_" + str(iteration) + "_CE"))
                    for net, params in zip(self.policy_estimators[policy["name"]].networks, torch.load(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "causal_estimator_networks_" + str(self.DATASET_PARAMS["intervention_name"]) + str(delta) + "_" + policy["model_type"] + "_" + policy["causal_type"] + "_" + str(iteration) + "_CE"))):
                        net.load_state_dict(params)
                        net.eval()
                    self.policy_estimators[policy["name"]].MODEL_PARAMS["seed"] = 99 + iteration*5

                performance, outcome_df, online_full_df, online_action_df = self.policy_online(self.EVALUATOR_PARAMS["nr_cases"], policy, iteration=iteration)

                self.online_performances[policy["name"]].append(performance)
                self.online_outcome_dfs[policy["name"]].append(outcome_df)
                self.online_full_dfs[policy["name"]].append(online_full_df)
                self.online_actions_dfs[policy["name"]].append(online_action_df)
                
                # NOTE: to change again

                # ESTIMATED PERFORMANCE
                offline_full_df, offline_actions_df = self.policy_offline(self.EVALUATOR_PARAMS["nr_cases"], self.bank_test_df, policy, iteration=iteration)

                self.offline_full_dfs[policy["name"]].append(offline_full_df)
                self.offline_actions_dfs[policy["name"]].append(offline_actions_df)
            
            # save

            # NOTE: to change again

            if self.BIG_EVAL:
                save_data(self.online_performances[policy["name"]], os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "online_performances_" + str(self.DATASET_PARAMS["intervention_name"]) + str(delta) + "_generator_iteration" + str(0) + self.suffix + str(policy["name"]) + str(self.evaluation_type_suffix)))
                save_data(self.online_outcome_dfs[policy["name"]], os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "online_outcome_dfs_" + str(self.DATASET_PARAMS["intervention_name"]) + str(delta) + "_generator_iteration" + str(0) + self.suffix + str(policy["name"]) + str(self.evaluation_type_suffix)))
                save_data(self.online_full_dfs[policy["name"]], os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "online_full_dfs_" + str(self.DATASET_PARAMS["intervention_name"]) + str(delta) + "_generator_iteration" + str(0) + self.suffix + str(policy["name"]) + str(self.evaluation_type_suffix)))
                save_data(self.online_actions_dfs[policy["name"]], os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "online_actions_dfs_" + str(self.DATASET_PARAMS["intervention_name"]) + str(delta) + "_generator_iteration" + str(0) + self.suffix + str(policy["name"]) + str(self.evaluation_type_suffix)))

                save_data(self.offline_full_dfs[policy["name"]], os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "offline_full_dfs_" + str(self.DATASET_PARAMS["intervention_name"]) + str(delta) + "_generator_iteration" + str(0) + self.suffix + str(policy["name"]) + str(self.evaluation_type_suffix)))
                save_data(self.offline_actions_dfs[policy["name"]], os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "offline_actions_dfs_" + str(self.DATASET_PARAMS["intervention_name"]) + str(delta) + "_generator_iteration" + str(0) + self.suffix + str(policy["name"]) + str(self.evaluation_type_suffix)))

        # NOTE: to change again

        # make sure that the offline df of the bank has the same cases as the offline dfs of the other policies
        case_nrs_other_policies = self.offline_full_dfs["random"][0]["case_nr"].unique()
        self.offline_full_dfs["bank"] = [self.offline_full_dfs["bank"][0][self.offline_full_dfs["bank"][0]["case_nr"].isin(case_nrs_other_policies)]]

        return self.online_performances, self.online_outcome_dfs, self.online_full_dfs, self.online_actions_dfs, self.offline_full_dfs, self.offline_actions_dfs
    
    def prep_offline_full_dfs(self, offline_full_dfs):
        prep_agg_offline_full_dict = {}
        prep_seq_offline_full_dict = {}
        for policy in offline_full_dfs.keys():
            delta = self.delta_policy
            # to get the offline dfs, the random policy is not influenced by the delta (confounding)
            if policy == "random" or policy == "bank":
                delta = 0.95
            # check if the file already exists (which is saved later on)
            # if self.BIG_EVAL:
            #     if "RealCause" in self.generator_classes.keys():
            #         if os.path.exists(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "prep_agg_offline_full_dict_eval_" + str(self.DATASET_PARAMS["intervention_name"]) + str(delta) + "_" + policy + "_generator_iteration" + str(0) + str(self.evaluation_type_suffix))) and not self.get_dfs_policies:
            #             prep_agg_offline_full_dict[policy] = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "prep_agg_offline_full_dict_eval_" + str(self.DATASET_PARAMS["intervention_name"])  + str(delta) + "_" + policy + "_generator_iteration" + str(0) + str(self.evaluation_type_suffix)))
            #             continue
            #     if "ProCause" in self.generator_classes.keys():
            #         if os.path.exists(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "prep_seq_offline_full_dict_eval_" + str(self.DATASET_PARAMS["intervention_name"]) + str(delta) + "_" + policy + "_generator_iteration" + str(0) + str(self.evaluation_type_suffix))) and not self.get_dfs_policies:
            #             prep_seq_offline_full_dict[policy] = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "prep_seq_offline_full_dict_eval_" + str(self.DATASET_PARAMS["intervention_name"]) +  str(delta) + "_" + policy + "_generator_iteration" + str(0) + str(self.evaluation_type_suffix)))
            #             continue

            # if policy == "bank":
            #     continue
            prep_agg_offline_full_dict[policy] = []
            prep_seq_offline_full_dict[policy] = []
            for iteration in range(self.EVALUATOR_PARAMS["num_iterations"]):
                if iteration >= len(offline_full_dfs[policy]):
                    continue
                offline_full_df = offline_full_dfs[policy][iteration]

                if "RealCause" in self.generator_classes.keys():
                    prep_agg_offline_full = self.case_preps["RealCause"].preprocess_eval_aggregated(offline_full_df, self.generator_classes["RealCause"].prep_utils)
                    prep_agg_offline_full_dict[policy].append(prep_agg_offline_full)
                if "ProCause" in self.generator_classes.keys():
                    prep_seq_offline_full = self.case_preps["ProCause"].preprocess_eval_sequential(offline_full_df, self.generator_classes["ProCause"].prep_utils) 
                    prep_seq_offline_full_dict[policy].append(prep_seq_offline_full)
            
            # save
            if self.BIG_EVAL:
                if "RealCause" in self.generator_classes.keys():
                    save_data(prep_agg_offline_full_dict[policy], os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "prep_agg_offline_full_dict_eval_" + str(self.DATASET_PARAMS["intervention_name"]) + str(delta) + "_" + policy + "_generator_iteration" + str(0) + str(self.evaluation_type_suffix)))
                if "ProCause" in self.generator_classes.keys():
                    save_data(prep_seq_offline_full_dict[policy], os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "prep_seq_offline_full_dict_eval_" + str(self.DATASET_PARAMS["intervention_name"]) + str(delta) + "_" + policy + "_generator_iteration" + str(0) + str(self.evaluation_type_suffix)))

        return prep_agg_offline_full_dict, prep_seq_offline_full_dict
    
    def get_estimated_outcomes(self, prep_agg_offline_full_dict, prep_seq_offline_full_dict, offline_actions_dfs):
        estimated_outcome_dfs_realcause = {}
        estimated_outcome_dfs_procause = {}

        # load the estimated outcome dfs if they exist
        # if "RealCause" in self.generator_classes.keys():
            # estimated_outcome_dfs_realcause = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "estimated_outcome_dfs_realcause_" + str(self.DATASET_PARAMS["intervention_name"]) + str(self.delta_evaluator) + "_generator_iteration" + str(self.generator_iteration) + "RealCause" + str(self.generator_classes["RealCause"].MODEL_PARAMS["causal_type"] + str(self.policy_names) + str(self.evaluation_type_suffix) + self.fixed_policy_delta_to_add)))
        # if "ProCause" in self.generator_classes.keys():
        #     estimated_outcome_dfs_procause = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "estimated_outcome_dfs_procause_" + str(self.DATASET_PARAMS["intervention_name"]) + str(self.delta_evaluator) + "_generator_iteration" + str(self.generator_iteration) + "ProCause" + str(self.generator_classes["ProCause"].MODEL_PARAMS["causal_type"] + str(self.policy_names) + str(self.evaluation_type_suffix) + self.fixed_policy_delta_to_add)))

        for policy in ["bank"] + list(offline_actions_dfs.keys()):
        # for policy in ["bank"]:
            estimated_outcome_dfs_realcause[policy] = []
            estimated_outcome_dfs_procause[policy] = []
            for iteration in range(self.EVALUATOR_PARAMS["num_iterations"]):
                if policy == "bank":
                    offline_actions_df = pd.DataFrame([])
                else:
                    offline_actions_df = offline_actions_dfs[policy][iteration]
                
                if "RealCause" in self.generator_classes.keys():
                    if iteration >= len(prep_agg_offline_full_dict[policy]):
                        continue
                    prep_agg_offline_full = prep_agg_offline_full_dict[policy][iteration]
                    outcome_df = self.policy_estimated_realcause(prep_agg_offline_full, offline_actions_df, iteration=iteration)
                    estimated_outcome_dfs_realcause[policy].append(outcome_df)
                if "ProCause" in self.generator_classes.keys():
                    if iteration >= len(prep_seq_offline_full_dict[policy]):
                        continue
                    prep_seq_offline_full = prep_seq_offline_full_dict[policy][iteration]
                    outcome_df = self.policy_estimated_procause(prep_seq_offline_full, offline_actions_df, iteration=iteration)
                    estimated_outcome_dfs_procause[policy].append(outcome_df)
        return estimated_outcome_dfs_realcause, estimated_outcome_dfs_procause
    
    def get_wssds(self, true_outcome_dfs, estimated_outcome_dfs_realcause, estimated_outcome_dfs_procause):
        wssd_dict_realcause = {}
        wssd_dict_procause = {}

        # check if the file already exists, if not save it, otherwise load it
        # if "RealCause" in self.generator_classes.keys():
        #     wssd_dict_realcause = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "wssd_dict_realcause_" + str(self.DATASET_PARAMS["intervention_name"]) + str(self.delta_evaluator) + "_generator_iteration" + str(self.generator_iteration) + "RealCause" + str(self.generator_classes["RealCause"].MODEL_PARAMS["causal_type"] + str(self.policy_names) + str(self.evaluation_type_suffix) + self.fixed_policy_delta_to_add)))
        # if "ProCause" in self.generator_classes.keys():
        #     wssd_dict_procause = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "wssd_dict_procause_" + str(self.DATASET_PARAMS["intervention_name"]) + str(self.delta_evaluator) + "_generator_iteration" + str(self.generator_iteration) + "ProCause" + str(self.generator_classes["ProCause"].MODEL_PARAMS["causal_type"] + str(self.policy_names) + str(self.evaluation_type_suffix) + self.fixed_policy_delta_to_add)))

        for policy in true_outcome_dfs.keys():
            if policy == "bank":
                continue
            wssd_dict_realcause[policy] = []
            wssd_dict_procause[policy] = []
            for iteration in range(self.EVALUATOR_PARAMS["num_iterations"]):
                if "RealCause" in self.generator_classes.keys():
                    wssd_realcause = self.calculate_wssd(true_outcome_dfs[policy][iteration], estimated_outcome_dfs_realcause[policy][iteration])
                    wssd_dict_realcause[policy].append(wssd_realcause)
                    print("WSSD RealCause: ", wssd_realcause, "for policy ", policy, '\n')
                    if self.BIG_EVAL:
                        save_data(wssd_realcause, os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "wssd_realcause_eval_iteration" + str(iteration) + "_" + str(self.DATASET_PARAMS["intervention_name"]) + self.generator_classes["RealCause"].MODEL_PARAMS["causal_type"] + str(self.delta_evaluator) + "_" + policy + "_generator_iteration" + str(self.generator_iteration) + str(self.evaluation_type_suffix) + self.fixed_policy_delta_to_add))
                if "ProCause" in self.generator_classes.keys():
                    wssd_procause = self.calculate_wssd(true_outcome_dfs[policy][iteration], estimated_outcome_dfs_procause[policy][iteration])
                    wssd_dict_procause[policy].append(wssd_procause)
                    print("WSSD ProCause: ", wssd_procause, "for policy ", policy, '\n')
                    if self.BIG_EVAL:
                        save_data(wssd_procause, os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "wssd_procause_eval_iteration" + str(iteration) + "_" + str(self.DATASET_PARAMS["intervention_name"]) + self.generator_classes["ProCause"].MODEL_PARAMS["causal_type"] + str(self.delta_evaluator) + "_" + policy + "_generator_iteration" + str(self.generator_iteration) + str(self.evaluation_type_suffix) + self.fixed_policy_delta_to_add))
        return wssd_dict_realcause, wssd_dict_procause
        
    def policy_online(self, n_cases, policy, iteration=0):
        print("Calculate True Performance for policy ", policy["name"])

        self.ExtraFlowConditioner = ExtraFlowConditioner()

        # INCLUDE: also save the online actions, and online full dfs
        policy_online_df = pd.DataFrame()
        policy_actions_df = pd.DataFrame()

        #Init performance metrics
        performance = 0
        outcome_df = pd.DataFrame()
        test_df = pd.DataFrame()

        #Init data generator
        case_gen = simulation.PresProcessGenerator(self.DATASET_PARAMS, seed=self.DATASET_PARAMS["random_seed_test"])

        if policy["name"] == "random":
            random_object_for_random_policy = random.Random(self.DATASET_PARAMS["random_seed_test"] + 5*iteration)
        
        #Run
        for case_nr in range(n_cases):
            if policy["name"] == "random":
                random_best_timing = get_random_best_timing(self.DATASET_PARAMS, 0, random_object_for_random_policy)
            if case_nr % 500 == 0 and case_nr != 0:
                    print("Case nr: ", case_nr)
                    print("Current performance", performance)
                    print('\n')
            current_case_outcomes = []
            best_action = 0
            for sample_nr in range(self.EVALUATOR_PARAMS["nr_samples_per_case"]):
                
                # NOTE: to change again
                
                # if sample_nr > 0:
                #     continue

                seed_to_add = case_nr if sample_nr == 0 else 0
                seed_to_add_conditioner = sample_nr*5
                prefix_list = []
                prefix_list = case_gen.start_simulation_inference(seed_to_add=seed_to_add, seed_to_add_conditioner=seed_to_add_conditioner)
                self.current_timing = 0
                while case_gen.int_points_available:
                    if self.current_timing % 2 == 0:
                        if policy["name"] == "bank":
                            best_action = get_bank_best_action(prefix_list, 0, self.DATASET_PARAMS)
                        elif policy["name"] == "random":
                            # NOTE: the sample number is extremely important here, because we want to have the same random action for each sample of the same case
                            if self.current_timing == random_best_timing and sample_nr == 0:
                                best_action = get_random_best_action(self.DATASET_PARAMS, 0, random_object_for_random_policy)
                        else:
                            best_action = get_causal_estimator_action(prefix_list[0], self.DATASET_PARAMS, self.policy_estimators[policy["name"]], self.opt_th, self.case_preps["estimator"])

                    # Break if intervention done or in last timing
                    prefix_list = case_gen.continue_simulation_inference(best_action)
                    if self.DATASET_PARAMS["intervention_info"]["name"] == ["time_contact_HQ"]:
                        self.current_timing += 1

                    if self.print_cases:
                        if policy["name"] == "random":
                            print("case_nr", case_nr)
                            print("best timing", random_best_timing)
                            print("Best action", best_action, '\n')

                full_case = case_gen.end_simulation_inference()
                full_case = pd.DataFrame(full_case)
                current_case_outcomes.append(full_case["outcome"].iloc[-1])
                
                if sample_nr == 0:
                    performance += full_case["outcome"].iloc[-1]
                    full_case["case_nr"] = case_nr
                    if policy["name"] == "bank":
                        test_df = pd.concat([test_df, full_case], axis=0)

                    if self.DATASET_PARAMS["intervention_info"]["name"] == ["time_contact_HQ"]:
                        if "start_priority" not in full_case["activity"].values:
                            # cut off to make a prefix: if 'contact_headquarters' is in the activities of the case, the prefix is up until 'contact_headquarters'
                            # if there is no 'contact_headquarters', the prefix is up until the LAST 'validate_application'
                            if "contact_headquarters" in full_case["activity"].values:
                                index_contact_headquarters = full_case[full_case["activity"] == "contact_headquarters"].index[0]
                                prefix_to_estimate = full_case[:index_contact_headquarters + 1]
                            else:
                                index_validate_application = full_case[full_case["activity"] == "validate_application"].index[-1]
                                prefix_to_estimate = full_case[:index_validate_application + 1]

                            action_df = pd.DataFrame({"case_nr": case_nr, "action": best_action}, index=[0])
                            policy_actions_df = pd.concat([policy_actions_df, action_df], axis=0, ignore_index=True)
                            policy_online_df = pd.concat([policy_online_df, prefix_to_estimate], axis=0, ignore_index=True)

                # if self.print_cases:
                #     print("Full case", full_case, "\n")
            
            # add to outcome_df with corresponding case_nr
            current_case_outcomes = pd.DataFrame(current_case_outcomes, columns=["outcome"])
            current_case_outcomes["case_nr"] = case_nr
            outcome_df = pd.concat([outcome_df, current_case_outcomes], axis=0, ignore_index=True)

        if policy["name"] == "bank":
            return performance, outcome_df, test_df, policy_online_df, policy_actions_df
        return performance, outcome_df, policy_online_df, policy_actions_df
    
    def policy_offline(self, n_cases, bank_test_df, policy, iteration=0):
        print("Get the offline dfs and actions for policy ", policy["name"])

        policy_offline_df = pd.DataFrame()
        policy_actions_df = pd.DataFrame()
        if policy["name"] == "random":
            random_object_for_random_policy = random.Random(self.DATASET_PARAMS["random_seed_test"] + 5*iteration)
        
        for case_nr in range(n_cases):
            if case_nr % 500 == 0 and case_nr != 0:
                print("Case nr: ", case_nr)
                print('\n')
            if policy["name"] == "random":
                random_best_timing = get_random_best_timing(self.DATASET_PARAMS, 0, random_object_for_random_policy)
            policy_treat_before_real_treat = False
            real_treat_happened = False
            both_never_treated = False
            just_take_true_outcome = False
            case = bank_test_df[bank_test_df["case_nr"] == case_nr]
            best_action = 0

            if self.DATASET_PARAMS["intervention_info"]["name"] == ["set_ir_3_levels"]:
                # check first whether the case has activity calculate_offer
                if "calculate_offer" not in case["activity"].values:
                    just_take_true_outcome = True
                else:
                    # prefix is up to calculate_offer (inclusive)
                    prefix_to_estimate = case[:case[case["activity"] == "calculate_offer"].index[0] + 1]
                    if policy["name"] == "bank":
                        best_action = get_bank_best_action(prefix_to_estimate, 0, self.DATASET_PARAMS)
                    elif policy["name"] == "random":
                        best_action = get_random_best_action(self.DATASET_PARAMS, 0, random_object_for_random_policy)
                    else:
                        best_action = get_causal_estimator_action(prefix_to_estimate, self.DATASET_PARAMS, self.policy_estimators[policy["name"]], self.opt_th, self.case_preps["estimator"])
                    policy_treat_before_real_treat = True
                    real_treat_happened = True

            timing = 0
            if self.DATASET_PARAMS["intervention_info"]["name"] == ["time_contact_HQ"]:
                if "start_priority" in case["activity"].values:
                    just_take_true_outcome = True
                else:
                    for index, row in case.iterrows():
                        if row["activity"] in self.DATASET_PARAMS["intervention_info"]["end_control_activity"][0]:
                            # make the prefix list with one more event (on top of the current row)
                            prefix_list = case[:index + 2]
                            if policy["name"] == "bank":
                                best_action = get_bank_best_action(prefix_list, 0, self.DATASET_PARAMS)
                            elif policy["name"] == "random":
                                if timing == random_best_timing:
                                    best_action = get_random_best_action(self.DATASET_PARAMS, 0, random_object_for_random_policy)
                            else:
                                best_action = get_causal_estimator_action(prefix_list, self.DATASET_PARAMS, self.policy_estimators[policy["name"]], self.opt_th, self.case_preps["estimator"])
                            
                            if best_action == 1:
                                policy_treat_before_real_treat = True
                                prefix_to_estimate = case[:index + 2]
                            
                            timing += 1 * 2
                        elif row["activity"] == "contact_headquarters":
                            real_treat_happened = True
                            prefix_to_estimate = case[:index + 1]
                        elif row["activity"] == "validate_application" and timing >= 8:
                            both_never_treated = True
                            prefix_to_estimate = case[:index + 1]

                        if policy_treat_before_real_treat or real_treat_happened or both_never_treated:
                            break
            
            if not just_take_true_outcome:
                action_df = pd.DataFrame({"case_nr": case_nr, "action": best_action}, index=[0])
                policy_actions_df = pd.concat([policy_actions_df, action_df], axis=0, ignore_index=True)
                policy_offline_df = pd.concat([policy_offline_df, prefix_to_estimate], axis=0, ignore_index=True)

        return policy_offline_df, policy_actions_df
    
    def policy_estimated_realcause(self, prep_agg_offline_full, action_df, iteration=0):
        seed = self.DATASET_PARAMS["random_seed_test"]
        # I want to return a df with nr_samples_per_case outcomes for each case nr

        total_outcome_df = pd.DataFrame()
        for sample in range(self.EVALUATOR_PARAMS["nr_samples_per_case"]):
            new_seed = seed + sample*5
            X = prep_agg_offline_full["X"]
            T = prep_agg_offline_full["T"]

            if action_df.empty:
                # get the case_nrs from the unique ones in the prep_agg_offline_full (but please note it is a tensor with 1 dimension)
                case_nrs = torch.unique(prep_agg_offline_full["case_nr"])
                outcome_df = pd.DataFrame({"case_nr": case_nrs, "outcome": None})
                y_ = self.generator_classes["RealCause"].sample_y(x=X, t=T, ret_counterfactuals=False, seed=new_seed)
            else:
                outcome_df = pd.DataFrame({"case_nr": action_df["case_nr"].values, "outcome": None})
                y_total_ = self.generator_classes["RealCause"].sample_y(x=X, t=T, ret_counterfactuals=True, seed=new_seed)
                y_total_stacked = torch.stack(y_total_)
                actions = torch.tensor(action_df["action"].values, dtype=torch.long)
                y_ = y_total_stacked[actions, torch.arange(y_total_stacked.shape[1])].unsqueeze(1)
                # y_ = y_total_[action_df["action"].values]
            
            y_pred = self.generator_classes["RealCause"].prep_utils["scaler_dict_train"]["outcome"].inverse_transform(y_.reshape(-1, 1))
            outcome_df["outcome"] = y_pred
            total_outcome_df = pd.concat([total_outcome_df, outcome_df], axis=0, ignore_index=True)
        return total_outcome_df
        # return outcome_df
    
    def policy_estimated_procause(self, prep_seq_offline_full, action_df, iteration=0):
        seed = self.DATASET_PARAMS["random_seed_test"]

        total_outcome_df = pd.DataFrame()
        for sample in range(self.EVALUATOR_PARAMS["nr_samples_per_case"]):
            new_seed = seed + sample*5
            X_case = prep_seq_offline_full["X_case"]
            X_event = prep_seq_offline_full["X_event"]
            prefix_len = prep_seq_offline_full["prefix_len"]
            T = prep_seq_offline_full["T"]

            if action_df.empty:
                # get the case_nrs from the unique ones in the prep_seq_offline_full (but please note it is a tensor with 1 dimension)
                case_nrs = torch.unique(prep_seq_offline_full["case_nr"])
                outcome_df = pd.DataFrame({"case_nr": case_nrs, "outcome": None})
                y_ = self.generator_classes["ProCause"].sample_y(x_case=X_case, x_event=X_event, t=T, prefix_len=prefix_len, ret_counterfactuals=False, seed=new_seed)
            else:
                outcome_df = pd.DataFrame({"case_nr": action_df["case_nr"].values, "outcome": None})
                y_total_ = self.generator_classes["ProCause"].sample_y(x_case=X_case, x_event=X_event, t=T, prefix_len=prefix_len, ret_counterfactuals=True, seed=new_seed)
                y_total_stacked = torch.stack(y_total_)
                actions = torch.tensor(action_df["action"].values, dtype=torch.long)
                y_ = y_total_stacked[actions, torch.arange(y_total_stacked.shape[1])].unsqueeze(1)
            # y_ = y_total_[action_df["action"].values]
            y_pred = self.generator_classes["ProCause"].prep_utils["scaler_dict_train"]["outcome"].inverse_transform(y_.reshape(-1, 1))
            outcome_df["outcome"] = y_pred
            total_outcome_df = pd.concat([total_outcome_df, outcome_df], axis=0, ignore_index=True)
        return total_outcome_df
        # return outcome_df
            
    def get_realcause_outcome(self, prefix, chosen_action):
        outcome_list = []
        prefix = pd.DataFrame(prefix)
        prep_prefix = self.case_preps["RealCause"].preprocess_sample_aggregated(prefix, self.generator_classes["RealCause"].prep_utils)
        outcome_scaler = self.generator_classes["RealCause"].prep_utils["scaler_dict_train"]["outcome"]
        x = prep_prefix["X"]
        t = prep_prefix["T"]
        for sample in range(self.EVALUATOR_PARAMS["nr_samples_per_case"]):
            seed = self.DATASET_PARAMS["random_seed_test"] + sample*5
            # generic for any number of outcomes
            y_total_ = self.generator_classes["RealCause"].sample_y(x=x, t=t, ret_counterfactuals=True, seed=seed)
            y_ = y_total_[chosen_action]
            y_pred = outcome_scaler.inverse_transform(y_.reshape(-1, 1))
            outcome_list.append(y_pred)
        # make sure outcome_list is just an array of outcomes, and not a list of an array of an array of outcomes (flatten twice)
        outcome_list = [outcome[0] for outcome in outcome_list]
        # once more
        outcome_list = [outcome[0] for outcome in outcome_list]
        return outcome_list
    
    def get_procause_outcome(self, prefix, chosen_action):
        outcome_list = []
        prefix = pd.DataFrame(prefix)
        prep_prefix = self.case_preps["ProCause"].preprocess_sample_sequential(prefix, self.generator_classes["ProCause"].prep_utils)
        outcome_scaler = self.generator_classes["ProCause"].prep_utils["scaler_dict_train"]["outcome"]
        x_case = prep_prefix["X_case"]
        x_event = prep_prefix["X_event"]
        prefix_len = prep_prefix["prefix_len"]
        t = prep_prefix["T"]
        for sample in range(self.EVALUATOR_PARAMS["nr_samples_per_case"]):
            seed = self.DATASET_PARAMS["random_seed_test"] + sample*5
            y_total_ = self.generator_classes["ProCause"].sample_y(x_case=x_case, x_event=x_event, t=t, prefix_len=prefix_len, ret_counterfactuals=True, seed=seed)
            y_ = y_total_[chosen_action]
            y_pred = outcome_scaler.inverse_transform(y_.reshape(-1, 1))
            outcome_list.append(y_pred)
        # make sure outcome_list is just an array of outcomes, and not a list of an array of an array of outcomes (flatten twice)
        outcome_list = [outcome[0] for outcome in outcome_list]
        # once more
        outcome_list = [outcome[0] for outcome in outcome_list]
        return outcome_list
    
    def calculate_wssd(self, true_outcome_df, estimated_outcome_df):
        # calculate the wasserstein distance for each case and take the average
        wssd_list = []
        # NOTE: estimated_outcome_df has nr_samples_per_case outcomes for each case_nr (but not in order, so case_nr 4 outcomes are spread out over the df)
        for case_nr in true_outcome_df["case_nr"].unique():
            if case_nr not in estimated_outcome_df["case_nr"].unique():
                wssd_list.append(0)
                continue
            true_outcomes = true_outcome_df[true_outcome_df["case_nr"] == case_nr]
            estimated_outcomes = estimated_outcome_df[estimated_outcome_df["case_nr"] == case_nr]
            wssd = stats.wasserstein_distance(true_outcomes["outcome"], estimated_outcomes["outcome"])
            wssd_list.append(wssd)
        return sum(wssd_list) / len(wssd_list)