from SimBank.activity_execution import ActivityExecutioner
import pandas as pd

def get_bank_best_action(prefix_list, current_int_index, DATASET_PARAMS):
        prefix_without_int = prefix_list[0][0:-1]
        prev_event = prefix_without_int[-1]
        action_index = 0
        
        if DATASET_PARAMS["intervention_info"]["name"][current_int_index] == "time_contact_HQ":
            cancel_condition = ((prev_event["unc_quality"] == 0 and prev_event["est_quality"] < DATASET_PARAMS["policies_info"]["min_quality"] and prev_event["noc"] >= DATASET_PARAMS["policies_info"]["max_noc"]) or (prev_event["noc"] >= DATASET_PARAMS["policies_info"]["max_noc"] and prev_event["unc_quality"] > 0))
            contact_condition = (prev_event["noc"] < 2 and prev_event["unc_quality"] == 0 and prev_event["amount"] > 10000 and prev_event["est_quality"] >= DATASET_PARAMS["policies_info"]["min_quality"])

            if cancel_condition:
                action_index = 0
            elif contact_condition:
                action_index = 1
        
        elif DATASET_PARAMS["intervention_info"]["name"][current_int_index] == "choose_procedure":
            priority_condition = (prev_event["amount"] > DATASET_PARAMS["policies_info"]["choose_procedure"]["amount"] and prev_event["est_quality"] >= DATASET_PARAMS["policies_info"]["choose_procedure"]["est_quality"])

            if priority_condition:
                action_index = 1
            else:
                action_index = 0
        
        elif DATASET_PARAMS["intervention_info"]["name"][current_int_index] == "set_ir_3_levels":
            activity_executioner = ActivityExecutioner()
            ir, _, _ = activity_executioner.calculate_offer(prev_event=prev_event, intervention_info=DATASET_PARAMS["intervention_info"])
            action_index = DATASET_PARAMS["intervention_info"]["actions"][current_int_index].index(ir)
        
        return action_index

def get_random_best_timing(DATASET_PARAMS, current_int_index, random_object_for_random_policy):
    if DATASET_PARAMS["intervention_info"]["name"] == ["time_contact_HQ"]:
        random_best_timing = random_object_for_random_policy.choice(range(DATASET_PARAMS["intervention_info"]["action_depth"][current_int_index] + 1)) * 2
    else:
        random_best_timing = 0
    return random_best_timing

def get_random_best_action(DATASET_PARAMS, current_int_index, random_object_for_random_policy):
    if DATASET_PARAMS["intervention_info"]["name"] == ["time_contact_HQ"]:
        # we make it inherently random by choosing the timing randomly (including timing of not doing the intervention)
        random_best_action = 1
    else:
        random_best_action = random_object_for_random_policy.choice(range(DATASET_PARAMS["intervention_info"]["action_width"][current_int_index]))
    return random_best_action

def get_causal_estimator_action(prefix, DATASET_PARAMS, estimator, opt_th, case_prep):
    prefix = pd.DataFrame(prefix)
    if estimator.MODEL_PARAMS["model_type"] in ["LSTM", "CNN", "LSTM-VAE"]:
        prep_prefix = case_prep.preprocess_sample_sequential(prefix, estimator.prep_utils)
        x_case = prep_prefix["X_case"]
        x_event = prep_prefix["X_event"]
        prefix_len = prep_prefix["prefix_len"]
        t = prep_prefix["T"]
        y_total_ = estimator.causal_functions.forward_seq(x_case=x_case, x_event=x_event, t=t, prefix_len=prefix_len, ret_counterfactuals=True)
        
    else:
        prep_prefix = case_prep.preprocess_sample_aggregated(prefix, estimator.prep_utils)
        x = prep_prefix["X"]
        t = prep_prefix["T"]
        y_total_ = estimator.causal_functions.forward_agg(x=x, t=t, ret_counterfactuals=True)
        
    if DATASET_PARAMS["action_depth"][0] > 1:
        diff_y = y_total_[1] - y_total_[0]
        if diff_y > opt_th:
            return 1
        return 0
    else:
        # return the index of the action with the highest outcome
        return y_total_.index(max(y_total_))