from src.utils.preprocessor.simbank_prep import SimBankPreprocessor
from src.utils.preprocessor.bpic_prep import BPICPreprocessor
import pandas as pd

class ProcessPreprocessor():
    def __init__(self, DATASET, raw_data, PREP_PARAMS, DATASET_PARAMS):
        self.DATASET = DATASET
        self.raw_data = raw_data
        self.PREP_PARAMS = PREP_PARAMS
        self.DATASET_PARAMS = DATASET_PARAMS

    def preprocess(self):
        if self.DATASET == "SimBank":
            self.data_ordered = self.order_data(self.raw_data)
            self.data_splitted_dict = self.split_data(self.data_ordered)
            preprocessor = SimBankPreprocessor(data_dict=self.data_splitted_dict, PREP_PARAMS=self.PREP_PARAMS, DATASET_PARAMS=self.DATASET_PARAMS)
            if self.PREP_PARAMS["encoding"] == "sequential":
                self.data_train_prep, self.data_val_prep, self.data_test_prep, self.prep_utils = preprocessor.preprocess_sequential()
            elif self.PREP_PARAMS["encoding"] == "aggregated":
                self.data_train_prep, self.data_val_prep, self.data_test_prep, self.prep_utils = preprocessor.preprocess_aggregated()
        
        elif "bpic" in self.DATASET:
            self.data_ordered = self.order_data(self.raw_data)
            self.data_splitted_dict = self.split_data(self.data_ordered)
            preprocessor = BPICPreprocessor(data_dict=self.data_splitted_dict, PREP_PARAMS=self.PREP_PARAMS, DATASET_PARAMS=self.DATASET_PARAMS)
            if self.PREP_PARAMS["encoding"] == "sequential":
                self.data_train_prep, self.data_val_prep, self.data_test_prep, self.prep_utils = preprocessor.preprocess_sequential()
            elif self.PREP_PARAMS["encoding"] == "aggregated":
                self.data_train_prep, self.data_val_prep, self.data_test_prep, self.prep_utils = preprocessor.preprocess_aggregated()

        return self.data_train_prep, self.data_val_prep, self.data_test_prep, self.prep_utils
            
    def order_data(self, data):
        if self.DATASET == "SimBank":
            data_ordered = data.sort_values(by=[self.DATASET_PARAMS["order_column"]])
        elif "bpic" in self.DATASET:
            data[self.DATASET_PARAMS["order_column"]] = pd.to_datetime(data[self.DATASET_PARAMS["order_column"]], utc=True, format="ISO8601")
            earliest_per_case = data.groupby(self.DATASET_PARAMS["case_nr_column"])[self.DATASET_PARAMS["order_column"]].min().reset_index()
            data = data.merge(earliest_per_case, on=self.DATASET_PARAMS["case_nr_column"], suffixes=('', '_case_min'))
            data_ordered = data.sort_values(by=['timestamp_case_min', self.DATASET_PARAMS["order_column"]]).drop(columns=['timestamp_case_min'])

            # renumber the case_nrs to be 0, 1, 2, ... instead of very large integers
            case_nrs = data_ordered[self.DATASET_PARAMS["case_nr_column"]].unique()
            case_nr_dict = {case_nr: i for i, case_nr in enumerate(case_nrs)}
            data_ordered[self.DATASET_PARAMS["case_nr_column"]] = data_ordered[self.DATASET_PARAMS["case_nr_column"]].map(case_nr_dict)

            # df = results_data2[0]
            # df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, format="ISO8601")
            # # sort by timestamp, but make sure the case_nrs are grouped together and sorted by timestamp, so that the first event of a case is the first row of that case, and is followed by the second event of that case, etc.
            # earliest_per_case = df.groupby(case_id_col)[timestamp_col].min().reset_index()
            # df = df.merge(earliest_per_case, on=case_id_col, suffixes=('', '_case_min'))
            # df_sorted = df.sort_values(by=['timestamp_case_min', timestamp_col]).drop(columns=['timestamp_case_min'])
            # results_data2[0] = df_sorted
        return data_ordered

    def split_data(self, data):
        # to change when using real data
        # split (per case though, and in the order of the data)
        case_nrs = data[self.DATASET_PARAMS["case_nr_column"]].unique()
        case_nrs_train = case_nrs[:int(len(case_nrs) * (1 - self.PREP_PARAMS["test_prop"] - self.PREP_PARAMS["val_prop"]))]
        case_nrs_val = case_nrs[int(len(case_nrs) * (1 - self.PREP_PARAMS["test_prop"] - self.PREP_PARAMS["val_prop"])):int(len(case_nrs) * (1 - self.PREP_PARAMS["test_prop"]))]
        case_nrs_test = case_nrs[int(len(case_nrs) * (1 - self.PREP_PARAMS["test_prop"])):]
        
        train_data = data[data[self.DATASET_PARAMS["case_nr_column"]].isin(case_nrs_train)]
        val_data = data[data[self.DATASET_PARAMS["case_nr_column"]].isin(case_nrs_val)]
        test_data = data[data[self.DATASET_PARAMS["case_nr_column"]].isin(case_nrs_test)]

        if "bpic" in self.DATASET:
            # get the max and min timestamp of the start of a case in each set (not overall max and min, but the max and min of the start of a case)
            max_start_timestamp_train = train_data.groupby(self.DATASET_PARAMS["case_nr_column"])[self.DATASET_PARAMS["order_column"]].min().max()
            min_start_timestamp_val = val_data.groupby(self.DATASET_PARAMS["case_nr_column"])[self.DATASET_PARAMS["order_column"]].min().min()
            max_start_timestamp_val = val_data.groupby(self.DATASET_PARAMS["case_nr_column"])[self.DATASET_PARAMS["order_column"]].min().max()
            min_start_timestamp_test = test_data.groupby(self.DATASET_PARAMS["case_nr_column"])[self.DATASET_PARAMS["order_column"]].min().min()
            assert max_start_timestamp_train < min_start_timestamp_val and max_start_timestamp_val < min_start_timestamp_test

            # cut_off cases that start after the start of the next set
            train_data = train_data[train_data[self.DATASET_PARAMS["order_column"]] <= min_start_timestamp_val]
            val_data = val_data[val_data[self.DATASET_PARAMS["order_column"]] <= min_start_timestamp_test]

        data_splitted_dict = {"train": train_data, "val": val_data, "test": test_data}
        return data_splitted_dict