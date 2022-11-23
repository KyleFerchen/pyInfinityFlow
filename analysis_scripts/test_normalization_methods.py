
print("Setting up environment...")
import pandas as pd
import numpy as np
import math
from scipy import stats
import anndata
import os
import gc

import sys
sys.path.append("/Users/kyleferchen/Documents/grimes_lab/tmp_store/tmp_cluster_scripts/development/")
from fcs_io import FCSFileObject
from scipy.stats import zscore
from scipy.interpolate import InterpolatedUnivariateSpline
import transformations
from sklearn.preprocessing import MinMaxScaler
import xgboost
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

import pickle

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import time


COMMON_LINEAR_FEATURES = ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W", "Time"]


# Helper functions
def read_fcs_into_anndata(fcs_file_path, obs_prefix="", batch_key=""):
    try:
        tmp_fcs = FCSFileObject(fcs_file_path=fcs_file_path)
        tmp_var_df = pd.DataFrame({"name": tmp_fcs.named_par,
                                   "USE_LOGICLE": [False if item in COMMON_LINEAR_FEATURES else True for item in tmp_fcs.named_par_channel],
                                   "LOGICLE_T": 3000000,
                                   "LOGICLE_W": 0,
                                   "LOGICLE_M": 3,
                                   "LOGICLE_A": 1,
                                   "LOGICLE_APPLIED": False,
                                   "IMPUTED": False},
                                    index = tmp_fcs.named_par_channel)
        tmp_obs_df = pd.DataFrame({"cell_number": list(range(tmp_fcs.data.shape[0]))},
                                    index = np.array([f'{obs_prefix}:{str(i)}' for i in list(range(tmp_fcs.data.shape[0]))]))
        if len(batch_key) > 0:
            tmp_obs_df['batch'] = batch_key
        tmp_anndata = anndata.AnnData(tmp_fcs.data.astype(np.float32), obs = tmp_obs_df, var = tmp_var_df)
        return(tmp_anndata)
    except Exception as e:
        print(e)
        print("ERROR! Failed to open file {}.".format(fcs_file_path))
        return(None)


def pprint(input_list):
    print("\n".join(input_list))


def anndata_to_df(input_anndata, use_raw_feature_names=True):
    tmp_feature_names = input_anndata.var.index.values
    if not use_raw_feature_names:
        try:
            tmp_feature_names = [f'{tmp_channel}:{tmp_name}' for tmp_channel, tmp_name in zip(tmp_feature_names, input_anndata.var['name'].values)]
        except:
            print("WARNING! Could not add feature names to channels. Make sure 'name' is a feature in input_anndata.var")
    return(pd.DataFrame(input_anndata.X, columns=tmp_feature_names, index=input_anndata.obs.index.values))


def scale_feature(input_array, min_threshold_percentile, max_threshold_percentile):
    input_array = np.array(list(input_array))
    min_threshold = np.percentile(input_array, min_threshold_percentile)
    max_threshold = np.percentile(input_array, max_threshold_percentile)
    input_array[input_array < min_threshold] = min_threshold
    input_array[input_array > max_threshold] = max_threshold
    return(MinMaxScaler().fit(input_array.reshape(-1,1)).transform(input_array.reshape(-1,1)).reshape(-1,))


def make_spline_function_for_reference(reference_signal, n=100):
    start_time = time.time()
    # n = 100    # Number of points to consider during integration of KDE model
    # Build the KDE model
    reference_kde_model = stats.gaussian_kde(reference_signal)
    tmp_total_area = reference_kde_model.integrate_box_1d(0, 1)
    tmp_percentiles = np.array([reference_kde_model.integrate_box(0, tmp_point) / tmp_total_area for tmp_point in np.linspace(0,1,n)])
    # Remove values out of range
    flag_out_of_range = (tmp_percentiles <= 0) | (tmp_percentiles >= 1)
    x_spline = tmp_percentiles[~flag_out_of_range]
    y_spline = np.linspace(0,1,n)[~flag_out_of_range]
    # Remove duplicates
    flag_duplicates = np.unique(x_spline, return_index=True)[-1]
    x_spline = x_spline[flag_duplicates]
    y_spline = y_spline[flag_duplicates]
    # Add boundaries
    x_spline = np.concatenate(([0], x_spline, [1]))
    y_spline = np.concatenate(([0], y_spline, [1]))
    reference_spline_function = InterpolatedUnivariateSpline(x_spline, y_spline, k=1, bbox=[0,1], ext=3)
    end_time = time.time()
    print("\tSpline Function Built after {:.2f} seconds.".format(end_time - start_time))
    return(reference_spline_function)


def map_input_to_reference_spline(input_signal, reference_spline_function):
    input_ranks = pd.Series(np.argsort(input_signal), index=list(range(len(input_signal)))).sort_values().index.values
    return(np.array([float(reference_spline_function(item)) for item in ((input_ranks + 1) / (len(input_signal)+2))]))




def apply_logicle_to_anndata(input_anndata, in_place=True):
    # Check which features require logicle transformation
    check_to_logicle = (input_anndata.var['USE_LOGICLE'] & ~input_anndata.var['LOGICLE_APPLIED']).values
    features_to_logicle = input_anndata.var.index.values[check_to_logicle]
    if len(features_to_logicle) == 0:
        print("WARNING! No features required logicle normalization at this time.")
        if in_place:
            return()
        else:
            return(input_anndata.copy())
    if not in_place:
        input_anndata = input_anndata.copy()
    try:
        input_anndata[:,features_to_logicle].X = anndata_to_df(input_anndata)\
            [features_to_logicle].apply(lambda x: transformations.apply_logicle(x.values, 
                T=input_anndata.var.loc[x.name, "LOGICLE_T"], 
                W=input_anndata.var.loc[x.name, "LOGICLE_W"], 
                M=input_anndata.var.loc[x.name, "LOGICLE_M"], 
                A=input_anndata.var.loc[x.name, "LOGICLE_A"]), axis=0).values
        input_anndata.var.loc[features_to_logicle, "LOGICLE_APPLIED"] = True
    except Exception as e:
        print("Failed to apply logicle function correctly!")
        print(e)
    if in_place:
        return()
    else:
        return(input_anndata)




# ## Normalize data between samples
# # Z-score, as in original Infinity Flow R Package
# t_start_zscore_1 = time.time()
# for i, infinity_marker in enumerate(ordered_markers):
#     tmp_indices = sub_t_adata.obs.loc[sub_t_adata.obs["batch"] == infinity_marker].index.values
#     tmp_backbone = file_handler.handles[infinity_marker]["backbone_channels"]
#     sub_t_adata[tmp_indices, tmp_backbone].X = zscore(sub_t_adata[tmp_indices, tmp_backbone].X.toarray())




print("Loading the data...")
# Define paths to the data
list_input_flow_files = ["/data/salomonis2/LabFiles/Kyle/Analysis/2022_07_01_Infinity_Flow_Update/input/gated_live_fcs_input/2021_12_13_PE/export_Infinity Flow OSU PE 2021_12-PE_Infinity_Markers-CD2_Unmixed_Live.fcs",
                         "/data/salomonis2/LabFiles/Kyle/Analysis/2022_07_01_Infinity_Flow_Update/input/gated_live_fcs_input/2021_12_13_PE/export_Infinity Flow OSU PE 2021_12-PE_Infinity_Markers-CD11a_Unmixed_Live.fcs",
                         "/data/salomonis2/LabFiles/Kyle/Analysis/2022_07_01_Infinity_Flow_Update/input/gated_live_fcs_input/2022_01_19_PE/export_Infinity Flow CCHMC PE 2022_01_19-PE_Infinity_Markers-CD1d_Unmixed_Live.fcs",
                         "/data/salomonis2/LabFiles/Kyle/Analysis/2022_07_01_Infinity_Flow_Update/input/gated_live_fcs_input/2022_01_19_PE/export_Infinity Flow CCHMC PE 2022_01_19-PE_Infinity_Markers-CD24_Unmixed_Live.fcs",
                         "/data/salomonis2/LabFiles/Kyle/Analysis/2022_07_01_Infinity_Flow_Update/input/gated_live_fcs_input/2022_05_20_PE/export_Infinity Flow 05-20-2022-Infinity Markers-CD14_Unmixed_Live.fcs",
                         "/data/salomonis2/LabFiles/Kyle/Analysis/2022_07_01_Infinity_Flow_Update/input/gated_live_fcs_input/2022_05_20_PE/export_Infinity Flow 05-20-2022-Infinity Markers-CD20_Unmixed_Live.fcs",
                         "/data/salomonis2/LabFiles/Kyle/Analysis/2022_07_01_Infinity_Flow_Update/input/gated_live_fcs_input/2022_06_28_PE/export_KF_Infinity_Flow_Add_PEs-Infinity_Markers_PE-TSLPR_Unmixed_Live.fcs",
                         "/data/salomonis2/LabFiles/Kyle/Analysis/2022_07_01_Infinity_Flow_Update/input/gated_live_fcs_input/2022_06_28_PE/export_KF_Infinity_Flow_Add_PEs-Infinity_Markers_PE-CD131_Unmixed_Live.fcs"]

# Batch (individual capture machine and time point) for each file
list_batch_info = [0,0,1,1,2,2,3,3]

# Replicate number
replicate_info = [1,2,1,2,1,2,1,2]

# The data from the fcs file
list_anndata = []
for tmp_file, tmp_batch, tmp_rep in zip(list_input_flow_files, list_batch_info, replicate_info):
    tmp_adata = read_fcs_into_anndata(fcs_file_path=tmp_file, obs_prefix=f"B{tmp_batch}R{tmp_rep}", batch_key=f"B{tmp_batch}R{tmp_rep}")
    list_anndata.append(tmp_adata[pd.Series(tmp_adata.obs.index.values).sample(50000, random_state=0).values,:])

print("Applying logicle normalization...")
combined = anndata.concat(list_anndata, merge='same')

apply_logicle_to_anndata(combined)

features_to_use = pd.Series(["APC-A",
                                "APC-Cy7-A",
                                "AlexaFluor647-A",
                                "AlexaFluor700-A",
                                "BUV395-A",
                                "BUV496-A",
                                "BUV661-A",
                                "BUV737-A",
                                "BUV805-A",
                                "BV421-A",
                                "BV480-A",
                                "BV510-A",
                                "BV570-A",
                                "BV605-A",
                                "BV650-A",
                                "BV711-A",
                                "BV750-A",
                                "GFP-A",
                                "PE-Cy5-A",
                                "PE-Cy7-A",
                                "V450-A"])


# Select 11 random features to use for training
training_features = features_to_use.sample(11, random_state=7).values

# Use remaining 10 features to test
testing_features = features_to_use[~features_to_use.isin(training_features).values].values


# # Define training set
# sel_training_batch = "B0R1"
# train_x = anndata_to_df(combined[combined.obs.loc[combined.obs["batch"] == sel_training_batch].index.values, training_features])
# tmp_y = testing_features[0]
# tmp_train_y = anndata_to_df(combined[combined.obs.loc[combined.obs["batch"] == sel_training_batch].index.values, [tmp_y]])
# # Train the model
# model = tmp_model = xgboost.XGBRegressor(n_jobs=12, random_state=7)
# model.fit(train_x.values, tmp_train_y.values)
# test_x = anndata_to_df(combined[:, training_features])
# tmp_predictions = pd.DataFrame({"true": combined[:,tmp_y].X.toarray().reshape(-1),
#                                 "pred": model.predict(test_x)},
#                                 index=combined.obs.index.values)
# tmp_predictions = tmp_predictions.loc[combined.obs.loc[combined.obs["batch"] != sel_training_batch].index.values]


# No normalization after logicle
print("Working on predictions without added normalization...")
predictions = {}
for sel_training_batch in combined.obs["batch"].unique():
    print(f"Working on batch {sel_training_batch}")
    train_x = anndata_to_df(combined[combined.obs.loc[combined.obs["batch"] == sel_training_batch].index.values, training_features])
    test_x = anndata_to_df(combined[combined.obs.loc[combined.obs["batch"] != sel_training_batch].index.values, training_features])
    predictions[sel_training_batch] = {}
    for tmp_y in testing_features:
        print(f"\tWorking on feature {tmp_y}...")
        tmp_train_y = anndata_to_df(combined[combined.obs.loc[combined.obs["batch"] == sel_training_batch].index.values, [tmp_y]])
        tmp_xgbstart = time.time()
        model = xgboost.XGBRegressor(n_jobs=12, random_state=7)
        model.fit(train_x.values, tmp_train_y.values)
        tmp_xgbend = time.time()
        print(f"\t\tTraining took {tmp_xgbend - tmp_xgbstart} seconds.\n\n")
        tmp_predictions = pd.DataFrame({"true": combined[test_x.index.values,tmp_y].X.toarray().reshape(-1),
                                        "pred": model.predict(test_x)},
                                        index=test_x.index.values)
        predictions[sel_training_batch][tmp_y] = tmp_predictions



# Z-score normalize data between each batch for the training columns
print("Working on z-score normalization for training columns...")
z_combined = combined.copy()
for sel_training_batch in z_combined.obs["batch"].unique():
    print(f"\tBatch {sel_training_batch}...")
    tmp_indices = z_combined.obs.loc[z_combined.obs["batch"] == sel_training_batch].index.values
    z_combined[tmp_indices, training_features].X = zscore(z_combined[tmp_indices, training_features].X.toarray(), axis=0)

print("Working on predictions after z-score normalization...")
z_predictions = {}
for sel_training_batch in z_combined.obs["batch"].unique():
    print(f"Working on batch {sel_training_batch}")
    train_x = anndata_to_df(z_combined[z_combined.obs.loc[z_combined.obs["batch"] == sel_training_batch].index.values, training_features])
    test_x = anndata_to_df(z_combined[z_combined.obs.loc[z_combined.obs["batch"] != sel_training_batch].index.values, training_features])
    z_predictions[sel_training_batch] = {}
    for tmp_y in testing_features:
        print(f"\tWorking on feature {tmp_y}...")
        tmp_train_y = anndata_to_df(z_combined[z_combined.obs.loc[z_combined.obs["batch"] == sel_training_batch].index.values, [tmp_y]])
        tmp_xgbstart = time.time()
        model = xgboost.XGBRegressor(n_jobs=12, random_state=7)
        model.fit(train_x.values, tmp_train_y.values)
        tmp_xgbend = time.time()
        print(f"\t\tTraining took {tmp_xgbend - tmp_xgbstart} seconds.\n\n")
        tmp_predictions = pd.DataFrame({"true": z_combined[test_x.index.values,tmp_y].X.toarray().reshape(-1),
                                        "pred": model.predict(test_x)},
                                        index=test_x.index.values)
        z_predictions[sel_training_batch][tmp_y] = tmp_predictions



# KDE distribution map normalize data using the first batch as the reference
print("Using KDE approximation to map distribution to reference...")
k_combined = combined.copy()
features_to_normalize = np.concatenate([training_features, testing_features])
batch_1_for_kde = anndata_to_df(k_combined[k_combined.obs.loc[k_combined.obs["batch"] == "B0R1"].index.values, features_to_normalize])
ref_functions = batch_1_for_kde.apply(lambda x: make_spline_function_for_reference(x.values), axis=0)
for tmp_batch in k_combined.obs["batch"].unique():
    if tmp_batch != "B0R1":
        print(f"\tBatch {tmp_batch}...")
        tmp_indices = k_combined.obs.loc[k_combined.obs["batch"] == tmp_batch].index.values
        tmp_data = anndata_to_df(k_combined[tmp_indices, features_to_normalize])
        k_combined[tmp_indices, features_to_normalize].X = tmp_data.apply(lambda x: map_input_to_reference_spline(x.values, ref_functions[x.name]), axis=0).values

print("Working on predictions after kde approximation normalization...")
k_predictions = {}
for sel_training_batch in k_combined.obs["batch"].unique():
    print(f"Working on batch {sel_training_batch}")
    train_x = anndata_to_df(k_combined[k_combined.obs.loc[k_combined.obs["batch"] == sel_training_batch].index.values, training_features])
    test_x = anndata_to_df(k_combined[k_combined.obs.loc[k_combined.obs["batch"] != sel_training_batch].index.values, training_features])
    k_predictions[sel_training_batch] = {}
    for tmp_y in testing_features:
        print(f"\tWorking on feature {tmp_y}...")
        tmp_train_y = anndata_to_df(k_combined[k_combined.obs.loc[k_combined.obs["batch"] == sel_training_batch].index.values, [tmp_y]])
        tmp_xgbstart = time.time()
        model = xgboost.XGBRegressor(n_jobs=12, random_state=7)
        model.fit(train_x.values, tmp_train_y.values)
        tmp_xgbend = time.time()
        print(f"\t\tTraining took {tmp_xgbend - tmp_xgbstart} seconds.\n\n")
        tmp_predictions = pd.DataFrame({"true": k_combined[test_x.index.values,tmp_y].X.toarray().reshape(-1),
                                        "pred": model.predict(test_x)},
                                        index=test_x.index.values)
        k_predictions[sel_training_batch][tmp_y] = tmp_predictions


print("Saving intermediate outputs...")

with open("test_normalization_int/base_predictions_dict.pickle", "wb") as tmp_file:
    pickle.dump(predictions, tmp_file)

with open("test_normalization_int/zscore_normalized_predictions_dict.pickle", "wb") as tmp_file:
    pickle.dump(z_predictions, tmp_file)
    
with open("test_normalization_int/kde_normalized_predictions_dict.pickle", "wb") as tmp_file:
    pickle.dump(k_predictions, tmp_file)

print("Done.")


def helper_build_error_tables(input_predictions, index_name):
    mean_abs_errors = {}
    mean_sq_errors = {}
    r2_values = {}
    for tmp_key in input_predictions:
        for tmp_marker in input_predictions[tmp_key]:
            tmp_true = input_predictions[tmp_key][tmp_marker].iloc[:,0].values
            tmp_pred = input_predictions[tmp_key][tmp_marker].iloc[:,1].values
            mean_abs_errors[f"{tmp_key}:{tmp_marker}"] = mean_absolute_error(tmp_true, tmp_pred)
            mean_sq_errors[f"{tmp_key}:{tmp_marker}"] = mean_squared_error(tmp_true, tmp_pred)
            r2_values[f"{tmp_key}:{tmp_marker}"] = r2_score(tmp_true, tmp_pred)
    mean_abs_errors = pd.DataFrame(mean_abs_errors, index=[index_name])
    mean_sq_errors = pd.DataFrame(mean_sq_errors, index=[index_name])
    r2_values = pd.DataFrame(r2_values, index=[index_name])
    return((mean_abs_errors, mean_sq_errors, r2_values))

mean_abs_errors, mean_sq_errors, r2_values = helper_build_error_tables(predictions, "base")
z_mean_abs_errors, z_mean_sq_errors, z_r2_values = helper_build_error_tables(z_predictions, "z-score")
k_mean_abs_errors, k_mean_sq_errors, k_r2_values = helper_build_error_tables(k_predictions, "kde")

mae_table = pd.concat([mean_abs_errors, z_mean_abs_errors, k_mean_abs_errors]).T
mse_table = pd.concat([mean_sq_errors, z_mean_sq_errors, k_mean_sq_errors]).T
r2_table = pd.concat([r2_values, z_r2_values, k_r2_values]).T

x = (mse_table["base"] - mse_table["kde"])
y = (mse_table["z-score"] - mse_table["kde"])

(mae_table["z-score"] - mae_table["kde"]).sum()

mse_table["base"] - mse_table["kde"]














