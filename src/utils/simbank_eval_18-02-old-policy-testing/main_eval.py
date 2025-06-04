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
    def __init__(self, DATASET_PARAMS, DATASET_PARAMS_INT, EVALUATOR_PARAMS, generator_classes, GENERATOR_PREP_PARAMS, ESTIMATOR_PARAMS, print_cases=False, delta=0.95, BIG_EVAL=True, generator_iteration=0):
        self.DATASET_PARAMS = DATASET_PARAMS
        self.DATASET_PARAMS_INT = DATASET_PARAMS_INT
        self.EVALUATOR_PARAMS = EVALUATOR_PARAMS
        self.generator_classes = generator_classes
        self.GENERATOR_PREP_PARAMS = GENERATOR_PREP_PARAMS
        self.ESTIMATOR_PARAMS = ESTIMATOR_PARAMS
        self.delta = delta
        self.print_cases = print_cases
        self.BIG_EVAL = BIG_EVAL
        self.generator_iteration = generator_iteration

        self.case_preps = {}
        if "RealCause" in self.GENERATOR_PREP_PARAMS.keys():
            self.case_preps["RealCause"] = simbank_prep.SimBankPreprocessor(DATASET_PARAMS=self.DATASET_PARAMS_INT, PREP_PARAMS=GENERATOR_PREP_PARAMS["RealCause"])
        if "ProCause" in self.GENERATOR_PREP_PARAMS.keys():
            self.case_preps["ProCause"] = simbank_prep.SimBankPreprocessor(DATASET_PARAMS=self.DATASET_PARAMS_INT, PREP_PARAMS=GENERATOR_PREP_PARAMS["ProCause"])
        
        self.case_preps["estimator"] = simbank_prep.SimBankPreprocessor(DATASET_PARAMS=self.DATASET_PARAMS_INT, PREP_PARAMS=self.ESTIMATOR_PARAMS["prep_params"])

    def evaluate(self):
        self.true_performances = {}
        self.true_outcome_dfs = {}
        self.true_full_dfs = {}
        self.realcause_outcome_dfs = {}
        self.procause_outcome_dfs = {}

        wssd_realcause_dict = {}
        wssd_procause_dict = {}

        # Get Bank policy performance, which is needed for the estimated performance
        self.bank_performance, self.bank_outcome_df, self.bank_test_df = self.policy_true(self.EVALUATOR_PARAMS["nr_cases"], {"name": "bank"})
        self.true_performances["bank"] = self.bank_performance
        self.true_outcome_dfs["bank"] = self.bank_outcome_df

        self.policy_estimators = {}
        # First get the true performance of each policy, then get the estimated performance (without bank policy)
        for policy in self.EVALUATOR_PARAMS["policies"]:
            self.true_performances[policy["name"]] = []
            self.true_outcome_dfs[policy["name"]] = []
            self.realcause_outcome_dfs[policy["name"]] = []
            self.procause_outcome_dfs[policy["name"]] = []

            wssd_realcause_dict[policy["name"]] = []
            wssd_procause_dict[policy["name"]] = []

            if policy["name"] not in ["random", "bank"]:
                self.ESTIMATOR_PARAMS["model_params"]["model_type"] = policy["model_type"]
                self.ESTIMATOR_PARAMS["model_params"]["causal_type"] = policy["causal_type"]
                self.ESTIMATOR_PARAMS["model_params"]["seed"] = 99
                causal_estimator = CausalEstimator(data_train=[], data_val=[], data_val_th=[], prep_utils=self.ESTIMATOR_PARAMS["prep_utils"][policy["model_type"]][0], MODEL_PARAMS=self.ESTIMATOR_PARAMS["model_params"], inference=True)
                self.policy_estimators[policy["name"]] = causal_estimator
                
            if policy["name"] == "bank":
                continue
            # TRUE PERFORMANCE
            for iteration in range(self.EVALUATOR_PARAMS["num_iterations"]):
                if policy["name"] not in ["bank", "random"]:
                    self.opt_th = 0
                    if self.DATASET_PARAMS["action_depth"][0] > 1:
                        self.opt_th = load_data(os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "opt_th_" + str(self.DATASET_PARAMS["intervention_name"]) + str(self.delta) + "_" + policy["model_type"] + "_" + policy["causal_type"] + "_" + str(iteration) + "_CE"))
                    for net, params in zip(self.policy_estimators[policy["name"]].networks, torch.load(os.path.join(os.getcwd(), RESULTS_FOLDER, "Simbank", "causal_estimator_networks_" + str(self.DATASET_PARAMS["intervention_name"]) + str(self.delta) + "_" + policy["model_type"] + "_" + policy["causal_type"] + "_" + str(iteration) + "_CE"))):
                        net.load_state_dict(params)
                        net.eval()
                    self.policy_estimators[policy["name"]].MODEL_PARAMS["seed"] = 99 + iteration*5

                performance, outcome_df, full_df = self.policy_true(self.EVALUATOR_PARAMS["nr_cases"], policy, iteration=iteration)

                self.true_performances[policy["name"]].append(performance)
                self.true_outcome_dfs[policy["name"]].append(outcome_df)
                self.true_full_dfs[policy["name"]] = full_df
                
                # self.true_performances[policy["name"]] = performance
                # self.true_outcome_dfs[policy["name"]] = outcome_df
                # self.true_full_dfs[policy["name"]] = full_df

                # ESTIMATED PERFORMANCE
                realcause_outcome_df, procause_outcome_df = self.policy_estimated(self.EVALUATOR_PARAMS["nr_cases"], self.bank_test_df, policy, iteration=iteration)

                self.realcause_outcome_dfs[policy["name"]].append(realcause_outcome_df)
                self.procause_outcome_dfs[policy["name"]].append(procause_outcome_df)

                # self.realcause_outcome_dfs[policy["name"]] = realcause_outcome_df
                # self.procause_outcome_dfs[policy["name"]] = procause_outcome_df
            
                # now calculate the wasserstein distance between the true outcome dfs and the estimated outcome dfs
                if "RealCause" in self.generator_classes.keys():
                    wssd_realcause = self.calculate_wssd(outcome_df, realcause_outcome_df)
                    # wssd_realcause_dict[policy["name"]] = wssd_realcause
                    wssd_realcause_dict[policy["name"]].append(wssd_realcause)
                    print("WSSD RealCause: ", wssd_realcause, "for policy ", policy["name"], '\n')
                    if self.BIG_EVAL:
                        save_data(wssd_realcause, os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "wssd_realcause_eval_iteration" + str(iteration) + "_" + str(self.DATASET_PARAMS["intervention_name"]) + self.generator_classes["RealCause"].MODEL_PARAMS["causal_type"] + str(self.delta) + "_" + policy["name"] + "_generator_iteration" + str(self.generator_iteration)))
                if "ProCause" in self.generator_classes.keys():
                    wssd_procause = self.calculate_wssd(outcome_df, procause_outcome_df)
                    # wssd_procause_dict[policy["name"]] = wssd_procause
                    wssd_procause_dict[policy["name"]].append(wssd_procause)
                    print("WSSD ProCause: ", wssd_procause, "for policy ", policy["name"], '\n')
                    if self.BIG_EVAL:
                        save_data(wssd_procause, os.path.join(os.getcwd(), RESULTS_FOLDER, "SimBank", "wssd_procause_eval_iteration" + str(iteration) + "_" + str(self.DATASET_PARAMS["intervention_name"]) + self.generator_classes["ProCause"].MODEL_PARAMS["causal_type"] + str(self.delta) + "_" + policy["name"] + "_generator_iteration" + str(self.generator_iteration)))

        return self.true_performances, self.true_outcome_dfs, self.realcause_outcome_dfs, self.procause_outcome_dfs, wssd_realcause_dict, wssd_procause_dict

    def policy_true(self, n_cases, policy, iteration=0):
        print("Calculate True Performance for policy ", policy["name"])

        self.ExtraFlowConditioner = ExtraFlowConditioner()

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
                    test_df = pd.concat([test_df, full_case], axis=0)

                # if self.print_cases:
                #     print("Full case", full_case, "\n")
            
            # add to outcome_df with corresponding case_nr
            current_case_outcomes = pd.DataFrame(current_case_outcomes, columns=["outcome"])
            current_case_outcomes["case_nr"] = case_nr
            outcome_df = pd.concat([outcome_df, current_case_outcomes], axis=0, ignore_index=True)
        
        return performance, outcome_df, test_df
    
    def policy_estimated(self, n_cases, bank_test_df, policy, iteration=0):
        print("Calculate Estimated Performance for policy ", policy["name"])

        realcause_outcome_df = pd.DataFrame()
        procause_outcome_df = pd.DataFrame()
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
            take_true_outcome = False
            case = bank_test_df[bank_test_df["case_nr"] == case_nr]
            best_action = 0

            if self.DATASET_PARAMS["intervention_info"]["name"] == ["set_ir_3_levels"]:
                # check first whether the case has activity calculate_offer
                if "calculate_offer" not in case["activity"].values:
                    take_true_outcome = True
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
                    take_true_outcome = True
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
            
            if "RealCause" in self.generator_classes.keys():
                if take_true_outcome:
                    realcause_outcome_list = [case["outcome"].iloc[-1]] * self.EVALUATOR_PARAMS["nr_samples_per_case"]
                else:
                    # Get the estimated performance by RealCause
                    realcause_outcome_list = self.get_realcause_outcome(prefix_to_estimate, best_action)
                realcause_outcome_list = pd.DataFrame(realcause_outcome_list, columns=["outcome"])
                realcause_outcome_list["case_nr"] = case_nr
                realcause_outcome_df = pd.concat([realcause_outcome_df, realcause_outcome_list], axis=0, ignore_index=True)
            if "ProCause" in self.generator_classes.keys():
                if self.print_cases:
                    if policy["name"] == 'random':
                        print("case_nr", case_nr)
                        print("best timing", random_best_timing)
                        print("Best action", best_action, '\n')
                if take_true_outcome:
                    procause_outcome_list = [case["outcome"].iloc[-1]] * self.EVALUATOR_PARAMS["nr_samples_per_case"]
                else:
                    # Get the estimated performance by ProCause
                    procause_outcome_list = self.get_procause_outcome(prefix_to_estimate, best_action)
                procause_outcome_list = pd.DataFrame(procause_outcome_list, columns=["outcome"])
                procause_outcome_list["case_nr"] = case_nr
                procause_outcome_df = pd.concat([procause_outcome_df, procause_outcome_list], axis=0, ignore_index=True)
        
        return realcause_outcome_df, procause_outcome_df
                
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
        for case_nr in true_outcome_df["case_nr"].unique():
            true_outcomes = true_outcome_df[true_outcome_df["case_nr"] == case_nr]
            estimated_outcomes = estimated_outcome_df[estimated_outcome_df["case_nr"] == case_nr]
            wssd = stats.wasserstein_distance(true_outcomes["outcome"], estimated_outcomes["outcome"])
            wssd_list.append(wssd)
        return sum(wssd_list) / len(wssd_list)