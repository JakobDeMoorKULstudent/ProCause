import torch
import src.utils.preprocessor.utils as utils
import pandas as pd
import numpy as np

class BPICPreprocessor():
    def __init__(self, PREP_PARAMS, DATASET_PARAMS, data_dict=None):
        self.data_dict = data_dict
        self.PREP_PARAMS = PREP_PARAMS
        self.DATASET_PARAMS = DATASET_PARAMS
        
        if self.data_dict is not None:
            self.data_train = self.data_dict['train']
            self.data_val = self.data_dict['val']
            self.data_test = self.data_dict['test']

            max_process_len_train = self.get_case_length_quantile(self.data_train)
            # Ref Teinemaa, 2019
            self.max_process_len = min(40, max_process_len_train)
            # self.max_process_len = 21
            # max_process_len_val = self.get_pos_case_length_quantile(self.data_val)
            # max_process_len_test = self.get_pos_case_length_quantile(self.data_test)
            # self.max_process_len = max(max_process_len_train, max_process_len_val, max_process_len_test)

        self.nr_treatment_columns = DATASET_PARAMS["action_width"][0] if DATASET_PARAMS["action_width"][0] > 2 else 1
        self.missing_value = -100

    def get_case_length_quantile(self, data, quantile=0.95):
        # Ref Teinemaa, 2019
        # if self.DATASET_PARAMS["dataset"] == "bpic2017":
        #     return 20
        # return int(np.ceil(data[data["outcome"]==0].groupby(["case_nr"]).size().quantile(quantile)))
        collection = data[data["treatment"]==1]["event_nr"]
        return int(np.ceil(collection.quantile(quantile)))

    """Aggregated preprocessing"""
    def preprocess_aggregated(self):
        # TRAIN
        # self.data_treat_train = self.add_treatment_column_aggregated(self.data_train)
        # self.data_pref_train = self.create_prefix_aggregations(self.data_treat_train)
        self.data_pref_train = self.create_prefix_aggregations(self.data_train)
        self.scale_cols_currently = self.DATASET_PARAMS["scale_cols"] + [col for col in self.data_pref_train.columns if "_count" in col] + ["prefix_len"]
        self.scaler_dict_train, self.data_scaled_train = utils.scale_columns(data = self.data_pref_train, scale_cols = self.scale_cols_currently)
        # handle missing values
        self.data_fill_train = self.handle_missing_values(data = self.data_scaled_train)
        T_array_train = np.array(self.data_fill_train["treatment"].tolist())
        if len(T_array_train.shape) == 1:
            T_array_train = T_array_train.reshape(-1, 1)
        self.data_train_prep = {
            "Y": torch.tensor(self.data_fill_train["outcome"].values.reshape(-1, 1), dtype=torch.float32),
            "T": torch.tensor(T_array_train, dtype=torch.float32),
            "X": torch.tensor(self.data_fill_train.drop(columns=["outcome", "treatment", "case_nr"]).values, dtype=torch.float32),
            "case_nr": torch.tensor(self.data_fill_train["case_nr"].values, dtype=torch.float32)
        }
        # T_array_train = np.array(self.data_scaled_train["treatment"].tolist())
        # if len(T_array_train.shape) == 1:
        #     T_array_train = T_array_train.reshape(-1, 1)
        # self.data_train_prep = {
        #     "Y": torch.tensor(self.data_scaled_train["outcome"].values.reshape(-1, 1), dtype=torch.float32),
        #     "T": torch.tensor(T_array_train, dtype=torch.float32),
        #     "X": torch.tensor(self.data_scaled_train.drop(columns=["outcome", "treatment", "case_nr"]).values, dtype=torch.float32),
        #     "case_nr": torch.tensor(self.data_scaled_train["case_nr"].values, dtype=torch.float32)
        # }

        # VAL
        # self.data_treat_val = self.add_treatment_column_aggregated(self.data_val)
        # self.data_pref_val = self.create_prefix_aggregations(self.data_treat_val)
        self.data_pref_val = self.create_prefix_aggregations(self.data_val, cols=self.data_pref_train.columns)
        _, self.data_scaled_val = utils.scale_columns(data = self.data_pref_val, scaler_dict = self.scaler_dict_train, scale_cols = self.scale_cols_currently)
        # handle missing values
        self.data_fill_val = self.handle_missing_values(data = self.data_scaled_val)
        T_array_val = np.array(self.data_fill_val["treatment"].tolist())
        if len(T_array_val.shape) == 1:
            T_array_val = T_array_val.reshape(-1, 1)
        self.data_val_prep = {
            "Y": torch.tensor(self.data_fill_val["outcome"].values.reshape(-1, 1), dtype=torch.float32),
            "T": torch.tensor(T_array_val, dtype=torch.float32),
            "X": torch.tensor(self.data_fill_val.drop(columns=["outcome", "treatment", "case_nr"]).values, dtype=torch.float32),
            "case_nr": torch.tensor(self.data_fill_val["case_nr"].values, dtype=torch.float32)
        }
        # T_array_val = np.array(self.data_scaled_val["treatment"].tolist())
        # if len(T_array_val.shape) == 1:
        #     T_array_val = T_array_val.reshape(-1, 1)
        # self.data_val_prep = {
        #     "Y": torch.tensor(self.data_scaled_val["outcome"].values.reshape(-1, 1), dtype=torch.float32),
        #     "T": torch.tensor(T_array_val, dtype=torch.float32),
        #     "X": torch.tensor(self.data_scaled_val.drop(columns=["outcome", "treatment", "case_nr"]).values, dtype=torch.float32),
        #     "case_nr": torch.tensor(self.data_scaled_val["case_nr"].values, dtype=torch.float32)
        # }

        # TEST
        if self.PREP_PARAMS["test_prop"] > 0:
            # self.data_treat_test = self.add_treatment_column_aggregated(self.data_test)
            # self.data_pref_test = self.create_prefix_aggregations(self.data_treat_test)
            self.data_pref_test = self.create_prefix_aggregations(self.data_test, cols=self.data_pref_train.columns)
            _, self.data_scaled_test = utils.scale_columns(data = self.data_pref_test, scaler_dict = self.scaler_dict_train, scale_cols = self.scale_cols_currently)
            # handle missing values
            self.data_fill_test = self.handle_missing_values(data = self.data_scaled_test)
            T_array_test = np.array(self.data_fill_test["treatment"].tolist())
            if len(T_array_test.shape) == 1:
                T_array_test = T_array_test.reshape(-1, 1)
            self.data_test_prep = {
                "Y": torch.tensor(self.data_fill_test["outcome"].values.reshape(-1, 1), dtype=torch.float32),
                "T": torch.tensor(T_array_test, dtype=torch.float32),
                "X": torch.tensor(self.data_fill_test.drop(columns=["outcome", "treatment", "case_nr"]).values, dtype=torch.float32),
                "case_nr": torch.tensor(self.data_fill_test["case_nr"].values, dtype=torch.float32)
            }
            # T_array_test = np.array(self.data_scaled_test["treatment"].tolist())
            # if len(T_array_test.shape) == 1:
            #     T_array_test = T_array_test.reshape(-1, 1)
            # self.data_test_prep = {
            #     "Y": torch.tensor(self.data_scaled_test["outcome"].values.reshape(-1, 1), dtype=torch.float32),
            #     "T": torch.tensor(T_array_test, dtype=torch.float32),
            #     "X": torch.tensor(self.data_scaled_test.drop(columns=["outcome", "treatment", "case_nr"]).values, dtype=torch.float32),
            #     "case_nr": torch.tensor(self.data_scaled_test["case_nr"].values, dtype=torch.float32)
            # }
        else:
            self.data_test_prep = {}
        
        self.prep_utils = {"scaler_dict_train": self.scaler_dict_train, "column_names": self.data_pref_train.columns, "scale_cols": self.scale_cols_currently}
    
        return self.data_train_prep, self.data_val_prep, self.data_test_prep, self.prep_utils
    
    def preprocess_eval_aggregated(self, data, prep_utils):
        data_treat = self.add_treatment_column_aggregated(data)
        data_pref = self.create_prefix_aggregations(data_treat, data_type="inference", cols=prep_utils["column_names"])
        _, data_scaled = utils.scale_columns(data = data_pref, scaler_dict = prep_utils["scaler_dict_train"], scale_cols = prep_utils["scale_cols"])
        T_array = np.array(data_scaled["treatment"].tolist())
        if len(T_array.shape) == 1:
            T_array = T_array.reshape(-1, 1)
        data_prep = {
            "Y": torch.tensor(data_scaled["outcome"].values.reshape(-1, 1), dtype=torch.float32),
            "T": torch.tensor(T_array, dtype=torch.float32),
            "X": torch.tensor(data_scaled.drop(columns=["outcome", "treatment", "case_nr"]).values, dtype=torch.float32),
            "case_nr": torch.tensor(data_scaled["case_nr"].values, dtype=torch.float32)
        }
        return data_prep
    
    def preprocess_sample_aggregated(self, sample, prep_utils):
        sample_treat = self.add_treatment_column_aggregated(sample)
        sample_pref = self.create_prefix_aggregations(sample_treat, data_type="inference", cols=prep_utils["column_names"])
        _, sample_scaled = utils.scale_columns(data = sample_pref, scaler_dict = prep_utils["scaler_dict_train"], scale_cols = prep_utils["scale_cols"])
        # sample_prep = {"Y": sample_scaled["outcome"].values, "T": sample_scaled["treatment"].values, "X": sample_scaled.drop(columns=["outcome", "treatment"]).values}
        T_array_sample = np.array(sample_scaled["treatment"].tolist())
        if len(T_array_sample.shape) == 1:
            T_array_sample = T_array_sample.reshape(-1, 1)
        sample_prep = {
            "Y": torch.tensor(sample_scaled["outcome"].values.reshape(-1, 1), dtype=torch.float32),
            "T": torch.tensor(T_array_sample, dtype=torch.float32),
            "X": torch.tensor(sample_scaled.drop(columns=["outcome", "treatment", "case_nr"]).values, dtype=torch.float32)
        }
        return sample_prep
    
    def create_prefix_aggregations(self, data, data_type="normal", cols=None):
        if cols is not None:
            activity_count_cols = [col for col in cols if "_count" in col]
        grouped_train = data.groupby("case_nr")
        self.scale_cols_aggregate = [col for col in self.DATASET_PARAMS["scale_cols"] if col not in self.PREP_PARAMS["last_state_cols"]]
        unique_activities = data["activity"].unique()
        unique_other_cat_cols_dict = {col: data[col].unique() for col in self.DATASET_PARAMS["cat_cols"] if col != "activity" and col not in self.PREP_PARAMS["last_state_cols"]}

        train_prep = pd.DataFrame()

        for case_nr, group in grouped_train:
            case_treated = False
            # check wether this is a treated case (with treatment == 1 in one of the rows), or a control case with treatment == 0 in all rows
            # control_case = group["treatment"].sum() == 0

            # initiate a mean for every scale aggregate column
            sum_dict = {col: 0 for col in self.scale_cols_aggregate}
            scale_col_count = {col: 0 for col in self.scale_cols_aggregate}
            # initiate a count for every activity
            activity_count_dict = {str(activity) + "_count": 0 for activity in unique_activities}
            # initiate a count for every other categorical column
            other_cat_col_dict_of_dicts = {col: {str(val) + "_count": 0 for val in unique_other_cat_cols_dict[col]} for col in unique_other_cat_cols_dict}
            # initiate a last state for every last state column
            last_state_dict = {col: 0 for col in self.PREP_PARAMS["last_state_cols"]}

            # if there is a row in the group with treatment == 1, only retain the the prefix ending at that row (not the previous ones)
            # if there is never a treatment == 1, retain all prefixes (for loop)
            for current_pos, (index, row) in enumerate(group.iterrows(), start=1):
                if case_treated:
                    break
                if current_pos > self.max_process_len:
                    break
                for col in self.scale_cols_aggregate:
                    if not pd.isna(row[col]):
                        sum_dict[col] += row[col]
                        scale_col_count[col] += 1
                activity_count_dict[row["activity"] + "_count"] += 1
                for col in unique_other_cat_cols_dict:
                    other_cat_col_dict_of_dicts[col][row[col] + "_count"] += 1

                # if row["treatment"] == 1:
                # check if row["treatment"] contains a 1 instead of row["treatment"] == 1
                if 1 in row["treatment"] if isinstance(row["treatment"], list) else row["treatment"] == 1:
                    for col in self.PREP_PARAMS["last_state_cols"]:
                        last_state_dict[col] = row[col]
                    
                    mean_dict = {col: ((sum_dict[col] / scale_col_count[col]) if scale_col_count[col] > 0 else 0) for col in self.scale_cols_aggregate}

                    # prefix = pd.DataFrame({**mean_dict, **activity_count_dict, **last_state_dict, **}, index=[0])
                    prefix = pd.DataFrame({**mean_dict, **activity_count_dict, **last_state_dict, **{k: v for d in other_cat_col_dict_of_dicts.values() for k, v in d.items()}}, index=[0])
                    prefix["treatment"] = [row["treatment"]]
                    prefix["prefix_len"] = current_pos
                    prefix["case_nr"] = [case_nr]
                    if data_type =="inference":
                        # if there is a col in activity_count_cols that is not in the prefix, add it with value 0
                        for col in activity_count_cols:
                            if col not in prefix.columns:
                                prefix[col] = 0
                    train_prep = pd.concat([train_prep, prefix], axis=0, ignore_index=True)
                    case_treated = True
                
                # if control_case:
                # if datatyp is inference, we just want to retain if the current row is the second last row of the case
                else:
                    inference_condition = data_type == "inference" and current_pos == len(group) - 1

                    if inference_condition or data_type != "inference":
                        for col in self.PREP_PARAMS["last_state_cols"]:
                            last_state_dict[col] = row[col]
                        
                        mean_dict = {col: ((sum_dict[col] / scale_col_count[col]) if scale_col_count[col] > 0 else 0) for col in self.scale_cols_aggregate}

                        # prefix = pd.DataFrame({**mean_dict, **activity_count_dict, **last_state_dict}, index=[0])
                        prefix = pd.DataFrame({**mean_dict, **activity_count_dict, **last_state_dict, **{k: v for d in other_cat_col_dict_of_dicts.values() for k, v in d.items()}}, index=[0])
                        prefix["treatment"] = [row["treatment"]]
                        prefix["prefix_len"] = current_pos
                        prefix["case_nr"] = [case_nr]
                        if data_type =="inference":
                            # if there is a col in activity_count_cols that is not in the prefix, add it with value 0
                            for col in activity_count_cols:
                                if col not in prefix.columns:
                                    prefix[col] = 0
                        train_prep = pd.concat([train_prep, prefix], axis=0, ignore_index=True)

        # one_hot_encode if there are categorical columns that are also in last_state_cols
        if len([col for col in self.PREP_PARAMS["last_state_cols"] if col in self.DATASET_PARAMS["cat_cols"]]) > 0:
            train_prep = utils.one_hot_encode_columns(data = train_prep, cat_cols = [col for col in self.PREP_PARAMS["last_state_cols"] if col in self.DATASET_PARAMS["cat_cols"]], case_cols = ["case_nr"], event_cols = [], oh_encoder_dict=None)[1]

        # if there are columns which have 0 as value in all rows, drop them --> goes from 21 until 12 columns
        if self.PREP_PARAMS["filter_useless_cols"]:
            if data_type == "inference":
                # only get the columns in cols
                train_prep = train_prep[cols]
            else:
                if cols is not None:
                    # check first if there are columns in cols that are not in train_prep
                    cols_not_in_train_prep = [col for col in cols if col not in train_prep.columns]
                    if len(cols_not_in_train_prep) > 0:
                        # add these columns with value 0
                        for col in cols_not_in_train_prep:
                            train_prep[col] = 0
                    train_prep = train_prep[cols]
                else:
                    train_prep = train_prep.loc[:, (train_prep != 0).any(axis=0)]

        return train_prep
    
    """Sequential preprocessing"""
    def preprocess_sequential(self):
        #TRAIN
        self.oh_encoder_dict_train, self.data_encoded_train, self.case_cols_encoded, self.event_cols_encoded = utils.one_hot_encode_columns(data = self.data_train, cat_cols = self.DATASET_PARAMS["cat_cols"], case_cols = self.DATASET_PARAMS["case_cols"], event_cols = self.DATASET_PARAMS["event_cols"])
        self.scaler_dict_train, self.data_scaled_train = utils.scale_columns(data = self.data_encoded_train, scale_cols = self.DATASET_PARAMS["scale_cols"])
        self.data_fill_train = self.handle_missing_values(data = self.data_scaled_train)
        # self.data_treat_train = self.add_treatment_column_sequential(data = self.data_fill_train, scaler_dict_train=self.scaler_dict_train)
        # self.data_train_prep = self.create_prefix_tensors(data = self.data_treat_train, max_process_len = self.max_process_len, data_type="normal")
        self.data_train_prep = self.create_prefix_tensors(data = self.data_fill_train, max_process_len = self.max_process_len, data_type="normal")
        self.case_cols_encoded, self.event_cols_encoded = self.data_train_prep["case_cols_encoded"], self.data_train_prep["event_cols_encoded"]

        #VAL
        self.oh_encoder_dict_val, self.data_encoded_val, _, _ = utils.one_hot_encode_columns(data = self.data_val, oh_encoder_dict = self.oh_encoder_dict_train, cat_cols = self.DATASET_PARAMS["cat_cols"], case_cols = self.DATASET_PARAMS["case_cols"], event_cols = self.DATASET_PARAMS["event_cols"])
        self.scaler_dict_val, self.data_scaled_val = utils.scale_columns(data = self.data_encoded_val, scaler_dict = self.scaler_dict_train, scale_cols = self.DATASET_PARAMS["scale_cols"])
        self.data_fill_val = self.handle_missing_values(data = self.data_scaled_val)
        # self.data_treat_val = self.add_treatment_column_sequential(data = self.data_fill_val, scaler_dict_train=self.scaler_dict_train)
        # self.data_val_prep = self.create_prefix_tensors(data = self.data_treat_val, max_process_len = self.max_process_len, data_type="normal")
        self.data_val_prep = self.create_prefix_tensors(data = self.data_fill_val, max_process_len = self.max_process_len, data_type="normal")

        # TEST
        if self.PREP_PARAMS["test_prop"] > 0:
            self.oh_encoder_dict_test, self.data_encoded_test, _, _ = utils.one_hot_encode_columns(data = self.data_test, oh_encoder_dict = self.oh_encoder_dict_train, cat_cols = self.DATASET_PARAMS["cat_cols"], case_cols = self.DATASET_PARAMS["case_cols"], event_cols = self.DATASET_PARAMS["event_cols"])
            self.scaler_dict_test, self.data_scaled_test = utils.scale_columns(data = self.data_encoded_test, scaler_dict = self.scaler_dict_train, scale_cols = self.DATASET_PARAMS["scale_cols"])
            self.data_fill_test = self.handle_missing_values(data = self.data_scaled_test)
            # self.data_treat_test = self.add_treatment_column_sequential(data = self.data_fill_test, scaler_dict_train=self.scaler_dict_train)
            # self.data_test_prep = self.create_prefix_tensors(data = self.data_treat_test, max_process_len = self.max_process_len, data_type="normal")
            self.data_test_prep = self.create_prefix_tensors(data = self.data_fill_test, max_process_len = self.max_process_len, data_type="normal")
        else:
            self.data_test_prep = {}

        self.prep_utils = {"scaler_dict_train": self.scaler_dict_train, 
                            "oh_encoder_dict_train": self.oh_encoder_dict_train, 
                            "max_process_len": self.max_process_len,
                            "case_cols_encoded": self.case_cols_encoded,
                            "event_cols_encoded": self.event_cols_encoded}

        return self.data_train_prep, self.data_val_prep, self.data_test_prep, self.prep_utils
    
    def preprocess_eval_sequential(self, data, prep_utils):
        _, data_encoded, _, _ = utils.one_hot_encode_columns(data = data, oh_encoder_dict = prep_utils["oh_encoder_dict_train"], cat_cols = self.DATASET_PARAMS["cat_cols"], case_cols = self.DATASET_PARAMS["case_cols"], event_cols = self.DATASET_PARAMS["event_cols"])
        _, data_scaled = utils.scale_columns(data = data_encoded, scaler_dict = prep_utils["scaler_dict_train"], scale_cols = self.DATASET_PARAMS["scale_cols"])
        data_fill = self.handle_missing_values(data = data_scaled)
        data_treat = self.add_treatment_column_sequential(data = data_fill, scaler_dict_train=prep_utils["scaler_dict_train"])
        data_prep = self.create_prefix_tensors(data = data_treat, max_process_len = prep_utils["max_process_len"], case_cols_encoded=prep_utils["case_cols_encoded"], event_cols_encoded=prep_utils["event_cols_encoded"], data_type="inference_dataset")
        return data_prep
    
    def preprocess_sample_sequential(self, sample, prep_utils):
        _, sample_encoded, _, _ = utils.one_hot_encode_columns(data = sample, oh_encoder_dict = prep_utils["oh_encoder_dict_train"], cat_cols = self.DATASET_PARAMS["cat_cols"], case_cols = self.DATASET_PARAMS["case_cols"], event_cols = self.DATASET_PARAMS["event_cols"])
        _, sample_scaled = utils.scale_columns(data = sample_encoded, scaler_dict = prep_utils["scaler_dict_train"], scale_cols = self.DATASET_PARAMS["scale_cols"])
        sample_fill = self.handle_missing_values(data = sample_scaled)
        sample_treat = self.add_treatment_column_sequential(data = sample_fill, scaler_dict_train=prep_utils["scaler_dict_train"])
        sample_prep = self.create_prefix_tensors(data = sample_treat, max_process_len = prep_utils["max_process_len"], case_cols_encoded=prep_utils["case_cols_encoded"], event_cols_encoded=prep_utils["event_cols_encoded"], data_type="inference_sample")
        return sample_prep
    
    def scale_atoms(self, averages, stdevs, threshold, scaler_dict):
        atoms_scaled = pd.DataFrame(averages)
        atoms_scaled = scaler_dict["outcome"].transform(atoms_scaled.values.reshape(-1, 1)).flatten()
        stdev_scaler = scaler_dict["outcome"].scale_[0]
        stdev_atoms_scaled = [stdevs[i] / stdev_scaler for i in range(len(stdevs))]
        bin_width_scaled = abs(scaler_dict["outcome"].transform(np.array(threshold).reshape(-1, 1)).flatten()[0])
        return atoms_scaled, stdev_atoms_scaled, bin_width_scaled

    def handle_missing_values(self, data):
        #Only floats are missing normally
        # data.fillna(-100, inplace=True)
        data.fillna(self.missing_value, inplace=True)
        return data

    def create_prefix_tensors(self, data, max_process_len, case_cols_encoded=None, event_cols_encoded=None, data_type="normal"):
        if case_cols_encoded is None:
            case_cols_encoded = self.case_cols_encoded
        if event_cols_encoded is None:
            event_cols_encoded = self.event_cols_encoded

        # save the indices
        treated_indices = []
        control_indices = []
        
        previous_case = -1
        case_treated_condition = False
        inference_dataset_indices = []
        # control_row_nrs_current_case = []
        X_cols = ["case_nr", "prefix_len"] + case_cols_encoded + event_cols_encoded
        X = torch.zeros(size=(len(data), len(X_cols) + self.nr_treatment_columns, max_process_len))
        for row_nr, row in data.iterrows():
            current_case = row["case_nr"]
            if current_case != previous_case:
                if data_type == "inference_dataset" and row_nr > 0:
                    inference_dataset_indices.append(row_nr - 1 - 1) #NOTE, additional -1 to retain without intervention

                event_nr = 0
                previous_case = current_case
                # add control indices if previous case was not treated
                # if not case_treated_condition:
                #     control_indices += control_row_nrs_current_case
                case_treated_condition = False
                # control_row_nrs_current_case = []
            else:
                # if treated and the case is still the same
                if case_treated_condition:
                    # go to the next row (no need to go through the rest of this case)
                    continue
                # copy all previous prefixes
                event_nr += 1
                if event_nr >= max_process_len:
                    continue
                X[row_nr, :, 0:event_nr] = X[row_nr-1, :, 0:event_nr]
            # add an event
            X[row_nr, 0, event_nr] = current_case

            # Process variable-length treatment list
            treatment_list = row["treatment"]
            X[row_nr, 1:1 + self.nr_treatment_columns, event_nr] = torch.tensor(treatment_list, dtype=torch.float32)
            last_index = 1+self.nr_treatment_columns

            X[row_nr, last_index, 0:event_nr + 1] = event_nr + 1
            last_index += 1

            X[row_nr, last_index:last_index + len(case_cols_encoded), event_nr] = torch.tensor(row[case_cols_encoded].values.astype(float))
            last_index += len(case_cols_encoded)
            X[row_nr, last_index: last_index + len(event_cols_encoded), event_nr] = \
                torch.tensor(row[event_cols_encoded].values.astype(float))
            
            if data_type == "normal":
                case_treated_condition = False
                control_condition = False
                
                # TREATMENT RETAIN
                if X[row_nr, 1:1 + self.nr_treatment_columns, event_nr].sum() > 0:
                # if X[row_nr, 1:1 + self.nr_treatment_columns, event_nr] == 1:
                    case_treated_condition = True
                else:
                # CONTROL RETAIN
                    all_zero_condition = torch.all(X[row_nr, 1:1 + self.nr_treatment_columns, :] == 0)
                    control_condition = all_zero_condition
                    # if len(self.DATASET_PARAMS["intervention_info"]["start_control_activity"]) > 0:
                    #     start_control_condition = False
                    #     end_control_condition = False
                    #     for start_control in self.DATASET_PARAMS["intervention_info"]["start_control_activity"]:
                    #         if 'activity_' + start_control in event_cols_encoded:
                    #             start_control_index = event_cols_encoded.index("activity_" + start_control)
                    #             start_control_condition = torch.any(X[row_nr, last_index + start_control_index, :] == 1)
                    #             if start_control_condition:
                    #                 break
                    #     for end_control in self.DATASET_PARAMS["intervention_info"]["end_control_activity"]:
                    #         if 'activity_' + end_control in event_cols_encoded:
                    #             end_control_index = event_cols_encoded.index("activity_" + end_control)
                    #             end_control_condition = (X[row_nr, last_index + end_control_index, event_nr] == 1)
                    #             if end_control_condition:
                    #                 break
                    #     control_condition = all_zero_condition and start_control_condition and end_control_condition
                    # else:
                    #     control_condition = all_zero_condition
                
                if case_treated_condition:
                    treated_indices.append(row_nr)
                elif control_condition:
                    # save the indices of the prefixes that could be a control case, then if we go to the next case and we see there was never a treatment, we can add these indices to the control indices
                    control_indices.append(row_nr)
                    # control_row_nrs_current_case.append(row_nr)

        prefix_len = X[:, 1 + self.nr_treatment_columns, 0]
        treatment = X[:, 1:1 + self.nr_treatment_columns, :]
        
        # Retaining
        treated_indices = torch.tensor(treated_indices)
        control_indices = torch.tensor(control_indices)
        if data_type == "inference_sample":
            last_index = prefix_len.size(0) - 1 - 1 #NOTE, additional -1 to retain without intervention
            retain_idx = torch.isin(torch.arange(treatment.size(0)), (last_index))
        elif data_type == "inference_dataset":
            # retain for all cases just the last possible prefix, also don't forget to add the last prefix of the last case
            inference_dataset_indices.append(prefix_len.size(0) - 1 - 1)
            retain_idx = torch.isin(torch.arange(treatment.size(0)), torch.tensor(inference_dataset_indices))
            lol = 1
        else:
            retain_idx = torch.isin(torch.arange(treatment.size(0)), torch.cat((control_indices, treated_indices)))

        Y = torch.Tensor(data["outcome"].values)[retain_idx]
        case_nr = X[retain_idx, 0 ,0]
        # Make T so that if there is a True in T, than it is just True, otherwise False
        T = torch.any(treatment[retain_idx, :, :], dim=2)
        last_index = 1 + self.nr_treatment_columns
        prefix_len = prefix_len[retain_idx]
        last_index += 1
        # X_case = X[retain_idx, last_index:last_index + len(case_cols_encoded), 0] #, :]
        lol = X[retain_idx, last_index:last_index + len(case_cols_encoded), :]
        X_case = X[retain_idx, last_index:last_index + len(case_cols_encoded), (prefix_len - 1).long()] #, :] #NOTE, -1 to retain last state
        last_index += len(case_cols_encoded)
        X_process = X[retain_idx, last_index: last_index + len(event_cols_encoded), :] #, :]

        # in X_process, if there are any 'cols' with all zeros, remove them, goes from 17 --> 8 for time_contact HQ, 17 --> 10 for calculate_offer
        if self.PREP_PARAMS["filter_useless_cols"] and data_type == "normal":
            filter_mask = ((X_process == 0) | (X_process == self.missing_value)).all(dim=2).all(dim=0)
            event_cols_encoded = [col for i, col in enumerate(event_cols_encoded) if not filter_mask[i]]
            X_process = X_process[:, ~filter_mask, :]

        return {"Y": Y, "case_nr": case_nr, "T": T, "prefix_len": prefix_len, "X_case": X_case, "X_event": X_process, "case_cols_encoded": case_cols_encoded, "event_cols_encoded": event_cols_encoded}