import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split


def values_showing_missings(metadata_path):
    """
    Uses the {abc}_metadaten.csv file of the NAKO/GNC data to produce a dict or a list. This file is expected to be in
    the same folder as the input NAKO data file.
    In case of the list, all unique values that indicate missings in the NAKO missings encoding.
    In case of the dict, a dict with all column_names that have such missings indicators as dict-keys and those values,
    indicating the missings in the form of lists of values as dict-values is output.
    :param metadata_path: <Path to actual NAKO data file>, i.e. the normal >30k file or the >13k OGTT file.
                          This is NOT(!) the path/to/{abc}_metadaten.csv file!
    :return: List of all unique values that indicate a missing in the actual NAKO datafile that was input
    """
    import csv
    csv.field_size_limit(500 * 1024* 1024)

    basename = Path(metadata_path).stem
    parent_directory = Path(metadata_path).parent
    explanation_path = parent_directory / f"{basename}_characteristics.csv"
    #                f"{basename}_Metadaten.csv"  # f"{parent_directory}/{basename}_Metadaten.csv"
    # in Linux, the above line is equivalent to <explanation_path = f"{parent_directory}/{basename}_Metadaten.csv">

    # Load the CSV file
    with open(explanation_path, newline='', encoding="latin1") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')
        next(csvreader, None)  # skip the header (first row)

        # Initialize the dictionary to store variable names and missing values
        missings_dict = {}

        # Iterate through each row in the CSV file
        for row in csvreader:
            # Extract the variable name and missing value number
            variable_name = row[0]
            missing_value = row[2]

            if "(Missing)" in row[3]:
                # Check if the variable name already exists in the dictionary
                if variable_name in missings_dict:
                    # Append the missing value to the list of missing values
                    missings_dict[variable_name].append(missing_value)
                else:
                    # Create a new list with the missing value
                    missings_dict[variable_name] = [missing_value]

    # Flatten the lists of values into a single list to produce a list of only all the unique value in the dict
    all_values = [val for sublist in missings_dict.values() for val in sublist]

    # 去重
    unique_values = set(all_values)
    unique_values_list = list(unique_values)

    # print('unique_values_list', unique_values_list)   # Print the list of unique values
    #
    # print('missings_dict', missings_dict)    # Print the dictionary with all values, indicating missing, depending on each column

    return unique_values_list  # Optionally, this function can also return a nice dict called <missings_dict>


def replace_missing_values(df_to_repair, all_unique_values_list, print_result=False):
    """
    Different measurements (column names) have different values to indicate that a patient didn't participate in
    one of the tests. E.g., the hand strength measurements use the value 888 for <patient didn't participate>.
    But some other value to indicate a missing measurement is 8888 and so on.
    Sadly, it appears that the following are all the <values that indicate missings>:
    ['-8', '333333', '9', '8888', '4', '8', '-9', '999999', '7776', '555555', '8886', '888', '0', '111111', '88',
    '9999', '666666', '444444', '222222', '7775', '99', '7777']
    This includes common realistic measurement results like 4/8/9/0, which are columns, we don't want to remove.
    :return: Dataset without the numbers in given list -- but values of [-1, 10) are never removed currently!
    """

    print(f"Going to remove these values from all columns in the dataframe: {all_unique_values_list}")

    for i in range(len(all_unique_values_list)):
        if int(all_unique_values_list[i]) > 10 or int(all_unique_values_list[i]) <= -1:
            df_to_repair.replace(int(all_unique_values_list[i]), np.nan,
                              inplace=True)  # Replace the missing values with NaN
            # df_to_repair.replace(int(all_unique_values_list[i]), 0, inplace=True)  # Replace the missing values with NaN

    if print_result is True:
        print(df_to_repair)

    return df_to_repair


def diabetes_groups(ogtt0h, ogtt2h, which_definition="WHO"):
    """
    Warning: The table's oGTT values are in mmol/l
    
    Lena diabetes groups:  
    1.    manifester DM: Nüchternplasmaglukose (ogTT 0h) ≥ 125 mg/dL und/oder 2h-Plasmaglukose nach 75 g oGTT (oGTT 2h) ≥ 200 mg/dL 

    2.    Prädiabetes:
    a.    erhöhter Nüchternblutzucker (IFG: impaired fasting gycaemia):  Nüchternplasmaglukose 110 bis 125 mg/dL
    b.    gestörte Glukosetoleranz (IGT: impaired glucose tolerance): 2h-Plasmaglukose nach 75 g oGTT 140 bis 200 mg/dL

    3.    Normoglykämie: Nüchternplasmaglukose < 110 mg/dL und 2h-Plasmaglukose nach 75 g oGTT < 140 mg/dL
    
    Umrechnungsfaktor von mmol/L in mg/dL ist:
        1 mmol/L = 180.16 g/mol * 1 mmol * 1/10 * 1/dL = 0.018 g/dL = 18 mg/dL
    Rückrichtung für mg/dL in mmol/L:
        1 mg/dL = 1/18 mmol/L = 0.0555 mmol/L
        
    Diabetes classes (X := ogtt0h; Y := ogtt2h):
    [0]: oGTT 0h ≥ 125 mg/dl  OR  oGTT 2h ≥ 200 mg/dl               // manifester DM
    ->   X ≥ 6.94 mmol/l      OR  Y ≥ 11.11 mmol/l
    [1]: 110 mg/dl ≤ oGTT 0h < 125 mg/dl  ->  6.11 ≤ X < 6.94       // Prädiabetes: erhöhter Nüchternblutzucker (IFG)
    [2]: 140 mg/dl ≤ oGTT 2h < 200 mg/dl  ->  7.78 ≤ Y < 11.11      // Prädiabetes: gestörte Glukosetoleranz (IGT)
    [3]: oGTT 0h < 110 mg/dl  AND  oGTT 2h < 140 mg/dl              // Normoglykämie
    ->   X < 6.11             AND  Y < 7.78

    Nachtrag, nach obiger "exakter" Umrechnung:
    Changed it to the values given by WHO diabetes diagnostic criteria:
    [0]: X ≥ 7 mmol/l  OR  Y ≥ 11.1 mmol/l          // Diabetes mellitus (DM)
    [1]: 6.1 ≤ X < 7                                // IFG
    [2]  7.8 ≤ Y < 11.1                             // IGT
    [3]: X < 6.1  AND  Y < 7.8                      // Normal

    Sources for the latter definition:
    1.) Definition and diagnosis of diabetes mellitus and intermediate hyperglycemia: Report of a WHO/IDF
    consultation (PDF). Geneva: World Health Organization. 2006. p. 21. ISBN 978-92-4-159493-6.
    2.)  Vijan S (March 2010). "In the clinic. Type 2 diabetes". Annals of Internal Medicine. 152 (5): ITC31-15, quiz
    ITC316. doi:10.7326/0003-4819-152-5-201003020-01003. PMID 20194231.

    :return: Diabetes classes instead of oGTT measurements (array of {0,1,2,3} instead of array of positive real numbers)
    """

    if not np.max(ogtt0h) < 100 or not np.max(ogtt2h) < 100:  # By now, all "missings" indicator values should be gone
        assert False, (f"The oGTT values don't make sense. Did the filtering fail? Are there still any indicator "
                       f"values for 'missings' included in the data? \n "
                       f"The maximums are {np.max(ogtt0h) = } and {np.max(ogtt2h) = } \n"
                       f"Reminder: The 'missings' indicators are 111111, 222222, 333333, 4444444, 555555, 666666 and"
                       f"999999.")

    # if ogtt0h.isnull().any() or ogtt2h.isnull().any():
    #     assert False, "There is a nan in the ogtt0h or ogtt2h data"

    diabetes_classes_array = np.array([])  # empty np.array

    if which_definition not in "WHO":
        # The following were my calculation results
        print(f"Applying own conversion results of Lena's mg/dl oGTT0h/oGTT2h class limits")
        for k in range(len(ogtt0h)):
            # Diabetes classes as four classes 0, 1, 2 and 3
            if ogtt0h.values[k] >= 6.94 or ogtt2h.values[k] >= 11.11:
                diabetes_classes_array = np.append(diabetes_classes_array, values=0)  # Class 0
            elif 6.11 <= ogtt0h.values[k] < 6.94:
                diabetes_classes_array = np.append(diabetes_classes_array, values=1)  # Class 1
            elif 7.78 <= ogtt2h.values[k] < 11.11:
                diabetes_classes_array = np.append(diabetes_classes_array, values=2)  # Class 2
            elif ogtt0h.values[k] < 6.11 and ogtt2h.values[k] < 7.78:
                diabetes_classes_array = np.append(diabetes_classes_array, values=3)  # Class 3
            else:
                assert False, (f"All cases of oGTT values should be caught and a class (0, 1, 2 or 3) determined for "
                               f"them, with the current class definitions. We can't get here! Look for mistakes in the "
                               f"if statement. \n"
                               f"The problem occurred for ogtt0h = {ogtt0h.values[k]} and ogtt2h = {ogtt2h.values[k]}, "
                               f"while {k = }")

    else:
        # The following are the WHO diabetes diagnostic criteria values when using mmol/L as the unit
        print("Applying the WHO diabetes diagnostic criteria on oGTT 0h and oGTT 2h")
        for k in range(len(ogtt0h)):
            # Diabetes classes as four classes 0, 1, 2 and 3
            if ogtt0h.values[k] >= 7 or ogtt2h.values[k] >= 11.1:
                diabetes_classes_array = np.append(diabetes_classes_array, values=0)  # Class 0
            elif 6.1 <= ogtt0h.values[k] < 7:
                diabetes_classes_array = np.append(diabetes_classes_array, values=1)  # Class 1
            elif 7.8 <= ogtt2h.values[k] < 11.1:
                diabetes_classes_array = np.append(diabetes_classes_array, values=2)  # Class 2
            elif ogtt0h.values[k] < 6.1 and ogtt2h.values[k] < 7.8:
                diabetes_classes_array = np.append(diabetes_classes_array, values=3)  # Class 3
            else:
                assert False, (f"All cases of oGTT values should be caught and a class (0, 1, 2 or 3) determined for "
                               f"them, with the current class definitions. We can't get here! Look for mistakes in the "
                               f"if statement. \n"
                               f"The problem occurred for ogtt0h = {ogtt0h.values[k]} and ogtt2h = {ogtt2h.values[k]}, "
                               f"while {k = }")

    return diabetes_classes_array


def change_floats_for_regression_to_classes_for_classification_labels(y):
    """
    Use multiple times if the float targets are split into y_train, y_test and y_val
    :param y: Targets of a regression problem that should change to five classes 0/1/2/3/4
    :return: np.array of classes instead of the input float values. They are split according to the regions in <bins>
    """
    # # Define the bins and labels for each class
    # bins = [0, 2, 4, 6, 8, 9999999]  # Classes are 1: 0...2 || 2: 2...4 || 3: 4...6 || ...
    # labels = [0, 1, 2, 3, 4]
    #
    # # Create a sample list of values
    # values = y
    #
    # # Convert the values to categories based on the bins and labels
    # categories = np.digitize(values, bins=bins, right=True)
    # print(np.unique(categories))
    # class_labels = categories

    y = np.asarray(y)

    import matplotlib.pyplot as plt

    n, bins, patches = plt.hist(y)
    plt.show()

    for i in range(len(y)):
        if y[i] < np.min(y) * 2:
            y[i] = 0
        elif y[i] < np.mean(y) / 1.5 and y[i] >= np.mean(y) / 2:
            y[i] = 1
        elif np.mean(y) > y[i] >= np.mean(y) / 1.5:
            y[i] = 2
        elif np.mean(y) * 1.5 > y[i] >= np.mean(y):
            y[i] = 3
        else:
            y[i] = 4

    class_labels = y

    print(
        f"Changed float values to the following classes and their underlying distribution: {np.unique(class_labels, return_counts=True)}")

    # unique_classes, class_counts = np.unique(class_labels, return_counts=True)
    # print(f"Changed float values to the following classes and their underlying distribution:")
    # for j in range(len(unique_classes)):
    #     print(unique_classes[j], ": ", class_counts[j])

    return class_labels


def average_all_cols_of_df_starting_with_water_fat_ip_opp(df):
    """
    PyRadiomics delivered 16449 features for a single patient. Most of them are the same calculation, calculated for all
    four signals (water, fat, in-phase, out-of-phase). This function returns a new pandas dataframe
    :return: pd.DataFrame where same columns are averaged for the four signals. Should reduce columns/feats by about 1/4
    """

    # Initialize a dictionary to count the occurrences of each suffix
    suffix_counts = {}

    # Iterate over column names
    for col in df.columns:
        # Check if the column name starts with any of the specified prefixes
        if col.startswith(("water_", "fat_", "in_", "opp_")):
            # Extract the suffix by removing the prefix
            suffix = col.split("_", 1)[1]
            # Count the occurrences of the suffix
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1

    # Initialize an empty list to store suffixes that appear four or more times
    suffix_list = [suffix for suffix, count in suffix_counts.items() if count >= 4]

    print(f"Found {len(suffix_list)} suffixes. The first three are: {suffix_list[0]}, {suffix_list[1]} and "
          f"{suffix_list[2]}")

    # Initialize a dictionary to store averages
    avg_dict = {}

    # Initialize a drop list for dropping column_names later
    drop_list = []

    # Iterate over the suffix list
    for suffix in suffix_list:
        avg_list = []
        # Iterate over the prefixes
        for prefix in ["water_", "fat_", "in_", "opp_"]:
            col_name = f"{prefix}{suffix}"
            # Append the column to avg_list
            avg_list.append(df[col_name])
            drop_list.append(col_name)
        # Calculate the mean across all columns with the same suffix
        avg_value = np.mean(avg_list, axis=0)
        avg_dict[suffix] = avg_value
        # print(f"Added the averages of each patient {avg_value} \n (values came from the columns "
        #       f"{[col.name for col in avg_list]}) \n to df under the new column name {suffix}")

    # Concatenate the averages to create a DataFrame
    avg_df = pd.DataFrame(avg_dict)

    # Concatenate the original DataFrame with the averages DataFrame
    result_df = pd.concat([df, avg_df], axis=1)
    print(f"Dropping {len(drop_list)} columns, the first few are {drop_list[0]}, {drop_list[1]} and {drop_list[2]}")
    result_df = result_df.drop(drop_list, axis="columns")

    return result_df
    #
    # else:
    #     assert False, (f"Averaging each feature over the 4 results for the 4 signals (fat/water/in-phase/out-of-phase) "
    #                    f"failed! It should never fail, unless no column starts with 'water', 'fat', 'in' or 'opp'."
    #                    f"Checking what prefix_columns became in the function: {prefix_columns = }"
    #                    f"This should not be empty when this function is used. If planned, set flag of data loading "
    #                    f"class method to False! I.e., \n"
    #                    f"<get_PyRad_feats_and_NAKO_diabetes_classes(average_feats_over_signals=False)>")


class getData():
    def __init__(self, diabetes_path=None):
        self.diabetes_path = diabetes_path
        print(f"Chosen path for >13k diabetes csv file: {self.diabetes_path}")

    def get_NAKO_diabetes_data(self, target_column_name="sa_ogtt0", drop_actual_diabetes_measurement_methods=False):

        diabetes_df = pd.read_csv(self.diabetes_path, sep=";", encoding="latin1")
        diabetes_df = diabetes_df.drop(["SE02"], axis=1)  # column with useless XML paths (useless strings)
        diabetes_df = diabetes_df.drop(["ID"], axis=1)  # let's drop the ID column as well

        # Helper functions, defined in the top of this .py file to change "missings" indicator values to np.nan
        list_of_unique_values_indicating_missings = values_showing_missings(self.diabetes_path)
        diabetes_df = replace_missing_values(diabetes_df, list_of_unique_values_indicating_missings)

        # In the diabetes data, the PWC130 values are in strings, e.g., "130 W" instead of the integer "130", therefore:
        diabetes_df['ergo_pwc130'] = diabetes_df['ergo_pwc130'].str.replace(' W', '')
        # # Convert the values to numeric types, replacing NaN values with -1
        diabetes_df['ergo_pwc130'] = pd.to_numeric(diabetes_df['ergo_pwc130'], errors='coerce').fillna(-1)
        diabetes_df['ergo_pwc130'] = diabetes_df['ergo_pwc130'].astype(int)
        diabetes_df['ergo_pwc130'] = diabetes_df['ergo_pwc130'].replace(-1, np.nan)

        # Calculate the average of the three left hand strength measurements
        diabetes_df['avg_lh_strength'] = diabetes_df[['hgr_lh1_kraft', 'hgr_lh2_kraft', 'hgr_lh3_kraft']].mean(axis=1)
        diabetes_df['avg_rh_strength'] = diabetes_df[['hgr_rh1_kraft', 'hgr_rh2_kraft', 'hgr_rh3_kraft']].mean(axis=1)
        # and drop the individual values afterward
        diabetes_df = diabetes_df.drop(['hgr_lh1_kraft', 'hgr_lh2_kraft', 'hgr_lh3_kraft',
                                        'hgr_rh1_kraft', 'hgr_rh2_kraft', 'hgr_rh3_kraft'], axis=1)

        diabetes_data = diabetes_df.loc[diabetes_df[target_column_name] < 100]  # This also removes nan from the target

        # Split the data into features (x) and labels (y)
        x = diabetes_data.drop(columns=[target_column_name], axis=1)
        y = diabetes_data[target_column_name]

        if drop_actual_diabetes_measurement_methods is True:
            try:
                x = x.drop(['sa_hba1c', 'sa_hba1c_plat', 'sa_hba1c_info', 'sa_hba1c_mat', 'sa_hba1c_meth',
                            'sa_hba1crel', 'sa_hba1crel_plat', 'sa_hba1crel_info', 'sa_hba1crel_mat',
                            'sa_hba1crel_meth',
                            'sa_ogtt0_plat', 'sa_ogtt0_info', 'sa_ogtt0_mat', 'sa_ogtt0_meth',
                            'sa_ogtt2_plat', 'sa_ogtt2_info', 'sa_ogtt2_mat', 'sa_ogtt2_meth'], axis=1)
            except Exception as e:
                print("Attention: \n"
                      "Tried to drop the following list of features from the input data (a pd.DataFrame): "
                      "['sa_hba1c', 'sa_hba1c_plat', 'sa_hba1c_info', 'sa_hba1c_mat', 'sa_hba1c_meth', "
                      "'sa_hba1crel', 'sa_hba1crel_plat', 'sa_hba1crel_info', 'sa_hba1crel_mat', 'sa_hba1crel_meth', "
                      "'sa_ogtt0_plat', 'sa_ogtt0_info', 'sa_ogtt0_mat', 'sa_ogtt0_meth', "
                      "'sa_ogtt2_plat', 'sa_ogtt2_info', 'sa_ogtt2_mat', 'sa_ogtt2_meth'])"
                      "but at least one wasn't found in the input features")
                print(f"Actual Error (which was caught) was: \n{e}")

        x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2)
        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2)  # val_size

        print(len(diabetes_df["avg_rh_strength"]))
        for i in range(len(diabetes_df["avg_rh_strength"])):
            if diabetes_df["avg_rh_strength"][i] > 100:
                print(i)
                print(f"OH NO! It's {diabetes_df['avg_rh_strength'][i]}")

        return x_train, y_train, x_test, y_test, x_val, y_val

    def get_NAKO_diabetes_classes(self, ogtt0_column_name="sa_ogtt0", ogtt2_column_name="sa_ogtt2",
                                  drop_actual_diabetes_measurement_methods=False, split_train_into_train_and_val=True):

        diabetes_df = pd.read_csv(self.diabetes_path, sep=";", encoding="latin1")
        diabetes_df = diabetes_df.drop(["SE02"], axis=1)  # column with useless XML paths (useless strings)
        diabetes_df = diabetes_df.drop(["ID"], axis=1)  # let's drop the ID column as well
        # diabetes_df = diabetes_df.drop(["basis_uort"], axis=1)  # Relevance of location would be a useless finding(?)

        # 频率编码:频率编码是一种将类别型变量转换为数值型变量的方法，
        quickly_check_frequency_encoding = True
        if quickly_check_frequency_encoding:
            from sklearn.preprocessing import LabelEncoder
            # Perform label encoding
            label_encoder = LabelEncoder()
            diabetes_df['basis_uort_encoded'] = label_encoder.fit_transform(diabetes_df['basis_uort'])
            # Drop the original 'basis_uort' column
            diabetes_df = diabetes_df.drop('basis_uort', axis=1)

        # Helper functions, defined in the top of this .py file to change "missings" indicator values to np.nan
        list_of_unique_values_indicating_missings = values_showing_missings(self.diabetes_path)
        diabetes_df = replace_missing_values(diabetes_df, list_of_unique_values_indicating_missings)

        # In the diabetes data, the PWC130 values are in strings, e.g., "130 W" instead of the integer "130", therefore:
        diabetes_df['ergo_pwc130'] = diabetes_df['ergo_pwc130'].str.replace(' W', '')
        # # Convert the values to numeric types, replacing NaN values with -1
        diabetes_df['ergo_pwc130'] = pd.to_numeric(diabetes_df['ergo_pwc130'], errors='coerce').fillna(-1)
        diabetes_df['ergo_pwc130'] = diabetes_df['ergo_pwc130'].astype(int)
        diabetes_df['ergo_pwc130'] = diabetes_df['ergo_pwc130'].replace(-1, np.nan)

        # Calculate the average of the three left hand strength measurements
        diabetes_df['avg_lh_strength'] = diabetes_df[['hgr_lh1_kraft', 'hgr_lh2_kraft', 'hgr_lh3_kraft']].mean(axis=1)
        diabetes_df['avg_rh_strength'] = diabetes_df[['hgr_rh1_kraft', 'hgr_rh2_kraft', 'hgr_rh3_kraft']].mean(axis=1)
        # and drop the individual values afterward
        diabetes_df = diabetes_df.drop(['hgr_lh1_kraft', 'hgr_lh2_kraft', 'hgr_lh3_kraft',
                                        'hgr_rh1_kraft', 'hgr_rh2_kraft', 'hgr_rh3_kraft'], axis=1)

        diabetes_df = diabetes_df.loc[diabetes_df[ogtt0_column_name] < 100]
        diabetes_data = diabetes_df.loc[diabetes_df[ogtt2_column_name] < 100]

        ogtt0_data = diabetes_data[ogtt0_column_name]
        ogtt2_data = diabetes_data[ogtt2_column_name]

        # Split the data into features (x) and labels (y)
        x = diabetes_data.drop(columns=[ogtt0_column_name, ogtt2_column_name], axis=1)
        y = diabetes_groups(ogtt0_data, ogtt2_data)

        # Optionally drop all input features, used by doctors to actually find out about the patient's diabetes status
        if drop_actual_diabetes_measurement_methods is True:
            try:
                x = x.drop(['sa_hba1c', 'sa_hba1c_plat', 'sa_hba1c_info', 'sa_hba1c_mat', 'sa_hba1c_meth',
                            'sa_hba1crel', 'sa_hba1crel_plat', 'sa_hba1crel_info', 'sa_hba1crel_mat',
                            'sa_hba1crel_meth',
                            'sa_ogtt0_plat', 'sa_ogtt0_info', 'sa_ogtt0_mat', 'sa_ogtt0_meth',
                            'sa_ogtt2_plat', 'sa_ogtt2_info', 'sa_ogtt2_mat', 'sa_ogtt2_meth'], axis=1)
            except Exception as e:
                print("Attention: \n "
                      "Tried to drop the following list of features from the input data (a pd.DataFrame): "
                      "['sa_hba1c', 'sa_hba1c_plat', 'sa_hba1c_info', 'sa_hba1c_mat', 'sa_hba1c_meth', "
                      "'sa_hba1crel', 'sa_hba1crel_plat', 'sa_hba1crel_info', 'sa_hba1crel_mat', 'sa_hba1crel_meth', "
                      "'sa_ogtt0_plat', 'sa_ogtt0_info', 'sa_ogtt0_mat', 'sa_ogtt0_meth', "
                      "'sa_ogtt2_plat', 'sa_ogtt2_info', 'sa_ogtt2_mat', 'sa_ogtt2_meth'])"
                      "but at least one wasn't found in the input features")
                print(f"Actual Error (which was caught) was: \n{e}")

        x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2)

        if split_train_into_train_and_val:
            x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2)
            return x_train, y_train, x_test, y_test, x_val, y_val

        else:
            print(f"Attention: x_val and y_val are returned as empty lists because {split_train_into_train_and_val = } "
                  f"was set to False by the user!")
            return x_train_val, y_train_val, x_test, y_test, [], []

    # def get_PyRad_feats_and_NAKO_diabetes_classes(self, ogtt0_column_name="sa_ogtt0", ogtt2_column_name="sa_ogtt2",
    #                                               average_feats_over_signals=True):
    #
    #     diabetes_df = pd.read_csv(self.diabetes_path, sep=";", encoding="latin1")
    #     diabetes_df = diabetes_df.drop(["SE02"], axis=1)  # column with useless XML paths (useless strings)
    #     # In this case, we drop the ID column after merging the pyrad and diabetes dataframes!!
    #
    #     # Helper functions, defined in the top of this .py file to change "missings" indicator values to np.nan
    #     list_of_unique_values_indicating_missings = values_showing_missings(self.diabetes_path)
    #     diabetes_df = replace_missing_values(diabetes_df, list_of_unique_values_indicating_missings)
    #
    #     # In the diabetes data, the PWC130 values are in strings, e.g., "130 W" instead of the integer "130", therefore:
    #     diabetes_df['ergo_pwc130'] = diabetes_df['ergo_pwc130'].str.replace(' W', '')
    #     # # Convert the values to numeric types, replacing NaN values with -1
    #     diabetes_df['ergo_pwc130'] = pd.to_numeric(diabetes_df['ergo_pwc130'], errors='coerce').fillna(-1)
    #     diabetes_df['ergo_pwc130'] = diabetes_df['ergo_pwc130'].astype(int)
    #     diabetes_df['ergo_pwc130'] = diabetes_df['ergo_pwc130'].replace(-1, np.nan)
    #
    #     diabetes_df = diabetes_df.loc[diabetes_df[ogtt0_column_name] < 100]
    #
    #     # Add the PyRadiomics features to this dataframe
    #     pyrad_df = pd.read_csv(self.pyrad_path, sep=",")
    #
    #     list_of_numeric_columns = list(pyrad_df.select_dtypes(include=[np.number]).columns.values)
    #     print(f"Found {len(list_of_numeric_columns)} numeric columns of the total {len(pyrad_df.columns) = }")
    #     pyrad_df = pyrad_df.select_dtypes(include=np.number)
    #
    #     if average_feats_over_signals:
    #         pyrad_df = average_all_cols_of_df_starting_with_water_fat_ip_opp(pyrad_df)
    #
    #     huge_merged_df = pd.merge(diabetes_df, pyrad_df, on="ID")
    #     huge_merged_df = huge_merged_df.drop(["ID"], axis=1)  # let's drop the ID column as well
    #
    #     diabetes_data = huge_merged_df.loc[huge_merged_df[ogtt2_column_name] < 100]
    #
    #     ogtt0_data = diabetes_data[ogtt0_column_name]
    #     ogtt2_data = diabetes_data[ogtt2_column_name]
    #
    #     # Split the data into features (x) and labels (y)
    #     x = diabetes_data.drop(columns=[ogtt0_column_name, ogtt2_column_name], axis=1)
    #     y = diabetes_groups(ogtt0_data, ogtt2_data)
    #
    #     x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2)
    #     x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2)
    #
    #     return x_train, y_train, x_test, y_test, x_val, y_val


if __name__ == "__main__":
    NAKO_data_object = getData()
    x_tr, y_tr, x_te, y_te, x_val, y_val = NAKO_data_object.get_NAKO_diabetes_data()

    a = 5

    # Test the function <replace_missing_values>
    # data = {
    #     'Patient ID': [1, 2, 3, 4, 5],
    #     'Measurement1': [0, 8888, 999999, 4, 5],
    #     'Measurement2': [-8, 2, 3, 4, 888],
    #     'Measurement3': [9999, 2, 3, 4, 5],
    # }
    #
    # df_to_repair = pd.DataFrame(data)
    # replace_missing_values(df_to_repair, [8888, -8, 999999, 5], print_result=True)
