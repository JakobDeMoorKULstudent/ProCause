from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np

def calculate_atoms(data, cutoff, group_size = 5):
    print("\nCALCULATING ATOMS\n")
    cutoff = cutoff
    # plot the outcome distribution per case_nr (group per case_nr and plot the outcome)
    max_outcomes = data.groupby('case_nr')['outcome'].max().reset_index()
    max_neg_outcomes = max_outcomes[max_outcomes["outcome"] <= cutoff]

    # Sort the DataFrame by 'values'
    df = max_neg_outcomes.sort_values(by='outcome').reset_index(drop=True)

    # Initialize variables
    groups = []
    current_group = []
    last_value = None
    threshold = 0

    # Grouping values within 100 of each other
    for value in df['outcome']:
        if last_value is None:
            last_value = value
            current_group.append(value)
        elif abs(value - last_value) <= threshold:
            current_group.append(value)
        else:
            groups.append(current_group)
            current_group = [value]
            last_value = value

    # Append the last group
    if current_group:
        groups.append(current_group)

    # Filter out groups that have fewer than 1 elements IMPORTANT NOTE: IMPORTANT TO KEEP THIS AT ONE, THEN THE BINS CAN BE BASED ON THE OUTCOME PER CASE_NR, OTHERWISE THE BINS SHOULD BE MADE AFTER RETAINING AND SCALING THE DATA
    filtered_groups = [group for group in groups if len(group) >= group_size]

    # Compute the average for each valid group
    grouped_averages = [sum(group) / len(group) for group in filtered_groups]

    # Also compute the standard deviation for each valid group
    grouped_stds = [np.std(group) for group in filtered_groups]

    #if a group has a standard dev lower than 1, than just take the most frequent value, and the stdev should be set to 0
    for index, std in enumerate(grouped_stds):
        if std < 1:
            grouped_averages[index] = max(set(filtered_groups[index]), key=filtered_groups[index].count)
            grouped_stds[index] = 0

    # Create a DataFrame to show the results
    result_df = pd.DataFrame(grouped_averages, columns=['Average'])

    print("Averages of the atoms: \n")
    print(result_df)
    # make a list of the averages
    averages = result_df['Average'].tolist()

    print("Standard deviations of those groups: \n")
    print(grouped_stds)

    split_list = [cutoff]   
    return averages, threshold, grouped_stds, split_list

def one_hot_encode_columns(data, cat_cols, case_cols, event_cols, oh_encoder_dict=None):
    case_cols_encoded = [col for col in case_cols if col not in cat_cols]
    event_cols_encoded = [col for col in event_cols if col not in cat_cols]

    if oh_encoder_dict is None:
        oh_encoder_dict = {}
        for col in cat_cols:
            oh_encoder_dict[col], data, cat_col_encoded = one_hot_encode_column(col = col, data = data)
            if col in case_cols:
                case_cols_encoded.extend(cat_col_encoded)
            elif col in event_cols:
                event_cols_encoded.extend(cat_col_encoded)
    else:
        #(oh_encoder_dict is known)
        for col, oh_encoder in oh_encoder_dict.items():
            _, data, cat_col_encoded = one_hot_encode_column(col = col, data = data, oh_encoder = oh_encoder)
            if col in case_cols:
                case_cols_encoded.extend(cat_col_encoded)
            elif col in event_cols:
                event_cols_encoded.extend(cat_col_encoded)

    return oh_encoder_dict, data, case_cols_encoded, event_cols_encoded
    
def one_hot_encode_column(col, data, oh_encoder=None):
    if oh_encoder is None:
        oh_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_col = oh_encoder.fit_transform(data[[col]])
        cat_col_encoded = oh_encoder.get_feature_names_out(input_features=[col])
    else:
        #(oh_encoder is known)
        encoded_col = oh_encoder.transform(data[[col]])
        cat_col_encoded = oh_encoder.get_feature_names_out(input_features=[col])
    df_enc = pd.DataFrame(encoded_col, columns=cat_col_encoded)
    data = data.reset_index(drop=True).join(df_enc)
    data.drop(columns=[col], inplace=True)

    return oh_encoder, data, cat_col_encoded

def scale_columns(data, scale_cols, scaler_dict=None):
    if scaler_dict is None:
        scaler_dict = {}
        for col in scale_cols:
            if col in data.columns:
                scaler_dict[col], data = scale_column(col, data)
    else:
        #(scale_dict is known)
        for col, scaler in scaler_dict.items():
            if col in data.columns:
                scale_column(col, data, scaler)

    return scaler_dict, data

def scale_column(col, data, scaler=None):
    #don't standardize missing values
    non_null_col_rows = ~data[col].isnull()

    if not data.loc[non_null_col_rows, col].empty:
        if scaler is None:
            scaler = StandardScaler()
            # make sure the dtype is correct
            data[col] = data[col].astype(float)
            data.loc[non_null_col_rows, col] = scaler.fit_transform(data.loc[non_null_col_rows, col].values.reshape(-1, 1)).flatten()
        else:
            #(scaler is known)
            # make sure the dtype is correct
            data[col] = data[col].astype(float)
            data.loc[non_null_col_rows, col] = scaler.transform(data.loc[non_null_col_rows, col].values.reshape(-1, 1)).flatten()
    return scaler, data