import argparse
import pandas as pd
import numpy as np
import scanpy as sc
import os
import pickle
import time
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)


from Debugging_Utilities import printv

from InfinityFlow_Utilities import read_annotation_table
from InfinityFlow_Utilities import anndata_to_df
from InfinityFlow_Utilities import check_infinity_flow_annotation_dataframes
from InfinityFlow_Utilities import setup_output_directories
from InfinityFlow_Utilities import single_chunk_training
from InfinityFlow_Utilities import single_chunk_testing
from InfinityFlow_Utilities import make_flow_regression_predictions
from InfinityFlow_Utilities import perform_background_correction
from InfinityFlow_Utilities import save_umap_figures_all_features
from InfinityFlow_Utilities import save_fcs_flow_anndata
from InfinityFlow_Utilities import move_features_to_silent
from InfinityFlow_Utilities import move_features_out_of_silent
from InfinityFlow_Utilities import make_pca_elbo_plot
from InfinityFlow_Utilities import find_markers_from_anndata

from Plotting_Utilities import assign_rainbow_colors_to_groups
from Plotting_Utilities import plot_leiden_clusters_over_umap

COMMON_LINEAR_FEATURES = ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W", "Time"]
# UMAP_PARAMS = {"n_neighbors":15, "min_dist":0.75, "metric":"correlation"}
UMAP_PARAMS = {}
XGB_PARAMS = {"n_estimators": 100, "learning_rate": 0.3}

def main():
    parser = argparse.ArgumentParser(description='User Defined Arguments')
    parser.add_argument('--data_dir', dest='data_dir', type=str, required=True,
        help='Directory with .fcs files with the same backbone channels to pool.')
    parser.add_argument('--out_dir', dest='output_dir', type=str, required=True,
        help='Directory to save the outputs.')
    parser.add_argument('--backbone_annotation', dest='backbone_annotation_file', 
        type=str, required=True,
        help='The .tsv or .csv file that tells which channels to use for the backbone.')
    parser.add_argument('--infinity_marker_annotation', 
        dest='infinity_marker_annotation_file', type=str, required=True,
        help='The .tsv or .csv file that tells which Infinity Markers correspond to which files.')
    parser.add_argument('--random_state', dest='random_state', type=int,
        help='Integer to specify the random_state, to make sampling, regression, and umap steps reproducible.',
        default=None)
    parser.add_argument('--use_logicle_scaling', dest='use_logicle_scaling', type=bool,
        help='Whether or not to apply logicle scaling to features that are typically '\
            'fluorescence channels, and not common linear features (Eg. FSC-A, SSC-A)',
        default=True)
    parser.add_argument('--normalization_method', dest='normalization_method', type=str,
        help='Method for normalizing feature values in an attempt to reduce sample to '\
            'sample batch effects. Options are:\n\tNone: no normalization\n\t'\
            'zscore: zscore normalize each of the backbone features (predictors)',
        default="zscore")
    parser.add_argument('--n_events_train', dest='n_events_train', type=int,
        help='Integer to specify the number events to use for training. 0 can be given for this '\
            '\nparameter and n_events_validate with the ratio_for_validation to use all events from'\
            '\nthe file.',
            default=0)
    parser.add_argument('--n_events_validate', dest='n_events_validate', type=int,
        help='Integer to specify the number events to use for validation. 0 can be given for this '\
            '\nparameter and n_events_train with the ratio_for_validation to use all events from'\
            '\nthe file.',
            default=0)
    parser.add_argument('--ratio_for_validation', dest='ratio_for_validation', type=float,
        help='(Float between 0 and 1) If n_events_train and n_events_validate are both 0, then all'\
            '\nof the events from the fcs file will be used and this parameter will '\
            '\nwill specify what percentage of the dataset should be used for validation.',
            default=0.2)
    parser.add_argument('--separate_backbone_reference', dest='separate_backbone_reference', type=str,
        help='The .fcs file passed as a file path string. This can be used as an alternative to \n'\
            '--n_events_combine. The regression will then be applied to the events in this file. \n'\
            'Each of the Infinity Markers specified in the infinity_marker_annotation file and \n'\
            'the original channel values for this separate_backbone_reference will be in the final output.',
            default=None)
    parser.add_argument('--n_events_combine', dest='n_events_combine', type=int,
        help='Integer to specify the number events from each file to pool into the final Infinity Flow object.',
        default=None)
    parser.add_argument('--n_final', dest='n_final', type=int,
        help='Integer to specify the final number of events to include for the final \n'\
            'Infinity Flow object. If n_final is 0, then all available events from \n'\
            'either the separate_backbone_reference or the combination or n_events_combine \n'\
            'from the InfinityMarker input files will be used.',
        default=0)
    parser.add_argument('--add_umap', dest='add_umap', type=bool,
        help='Boolean to specify if UMAP dimensionality reduction should be carried out on the '\
            'Infinity Flow object',
        default=False)
    parser.add_argument('--find_clusters', dest='find_clusters', type=bool,
        help='Boolean to specify if clustering should be done using Leiden clustering.',
        default=False)
    parser.add_argument('--find_markers', dest='find_markers', type=bool,
        help='Boolean to specify if MarkerFinder should be applied to find optimal '
            'markers for clusters.',
        default=False)
    parser.add_argument('--make_feature_plots', dest='make_feature_plots', type=bool,
        help='Boolean to specify if markers should be plotted over UMAP plots',
        default=False)
    parser.add_argument('--use_pca', dest='use_pca', type=bool,
        help='Boolean to specify if principal component ananlysis should be used to '\
            'reduce the feature space prior to UMAP and clustering. This is suggested '\
            'to save computation time.',
        default=True)
    parser.add_argument('--n_pc', dest='n_pc', type=int,
        help='Integer to specify the number principal components to use for UMAP.',
        default=15)
    parser.add_argument('--n_pc_plot_qc', dest='n_pc_plot_qc', type=int,
        help='Integer to specify the number principal components to plot in the elbo curve. '\
            'Helpful to estimate the number of principal components to use downstream.',
        default=50)
    parser.add_argument('--save_h5ad', dest='save_h5ad', type=bool,
        help='Boolean to specify if the Infinity Flow object should be saved as an h5ad file',
        default=False)
    parser.add_argument('--save_feather', dest='save_feather', type=bool,
        help='Save the Infinity Flow object as a feather file',
        default=False)
    parser.add_argument('--save_file_handler', dest='save_file_handler', type=bool,
        help='Save the file_handler object to <out_dir>/QC/file_handler.pickle',
        default=False)
    parser.add_argument('--save_regression_models', dest='save_regression_models', type=bool,
        help='Save the regression_models object to <out_dir>/QC/regression_models.pickle',
        default=False)
    parser.add_argument('--verbosity', dest='verbosity', type=bool,
        help='The level of verbosity with which to write to std-out. \n'\
            '0 = no print statements, to 3 = all debug print statements.',
        default=1)
    parser.add_argument('--n_cores', dest='n_cores', type=int,
        help='(int) to specify the number of cores to use.',
        default=1)
    args = parser.parse_args()
    try:
        fcs_file_dir = args.data_dir
        output_dir = args.output_dir
        backbone_annotation_file = args.backbone_annotation_file
        infinity_marker_annotation_file = args.infinity_marker_annotation_file
        RANDOM_STATE = args.random_state
        use_logicle_scaling = args.use_logicle_scaling
        normalization_method = args.normalization_method
        n_events_train = args.n_events_train
        n_events_validate = args.n_events_validate
        ratio_for_validation = args.ratio_for_validation
        separate_backbone_reference = args.separate_backbone_reference
        n_events_combine = args.n_events_combine
        n_final = args.n_final
        add_umap = args.add_umap
        find_clusters = args.find_clusters
        find_markers = args.find_markers
        make_feature_plots = args.make_feature_plots
        use_pca = args.use_pca
        n_pc = args.n_pc
        n_pc_plot_qc = args.n_pc_plot_qc
        save_h5ad = args.save_h5ad
        save_feather = args.save_feather
        save_file_handler = args.save_file_handler
        save_regression_models = args.save_regression_models
        VERBOSITY = args.verbosity
        cores_to_use = args.n_cores
    except Exception as e:
        printv(VERBOSITY, v3=str(e))
        print("Failed to parse arguments...")

    ### pyInfinityFlow_single_directory_constant_backbone -> function to replicate the R workflow in Python
    t_start_base_inflow_pipeline = time.time()
    printv(VERBOSITY, v1 = "\n\nRunning InfinityFlow from single directory, with all input .fcs files using the same channels for backbone...")
    printv(VERBOSITY, v2 = "\tInput directory: {}\n".format(fcs_file_dir))
    ## Initial checks
    # Read in the annotation files
    backbone_annotation = read_annotation_table(backbone_annotation_file)
    infinity_marker_annotation = read_annotation_table(infinity_marker_annotation_file)

    # Check if user wants to pool events or use a separate reference for the final backbone
    if separate_backbone_reference is not None:
        if n_events_combine is not None:
            raise ValueError("separate_backbone_reference and n_events_combine can not be "\
                "used together. One must be None.")
    else:
        if n_events_combine is None:
            raise ValueError("One of either separate_backbone_reference or n_events_combine "\
                "must not be None.")

    # Check if user wants to find_markers
    if find_markers:
        if not find_clusters:
            print("WARNING! find_markers was set to True but find_clusters was set"\
                "to False.\nfind_clusters must be set to True... Resetting.")
            find_clusters = True

    # Build file_handler object to set up how to use fcs files for training
    file_handler = check_infinity_flow_annotation_dataframes(backbone_annotation=backbone_annotation, 
        infinity_marker_annotation=infinity_marker_annotation,
        n_events_train=n_events_train, 
        n_events_validate=n_events_validate, 
        n_events_combine=n_events_combine, 
        ratio_for_validation=ratio_for_validation,
        separate_backbone_reference=separate_backbone_reference,
        random_state=RANDOM_STATE, 
        input_fcs_dir=fcs_file_dir, 
        verbosity=VERBOSITY)

    # Set up the output directories
    output_paths = setup_output_directories(output_dir = output_dir, 
        file_handler = file_handler, 
        verbosity=VERBOSITY)

    # Carry out the training, saving the regression_models as an object
    regression_models, timings_1 = single_chunk_training(file_handler = file_handler,
        random_state=RANDOM_STATE,
        cores_to_use=cores_to_use,
        xgb_params=XGB_PARAMS,
        use_logicle_scaling=use_logicle_scaling, 
        normalization_method=normalization_method,  
        verbosity=VERBOSITY)

    # Carry out validations of the model, saving metrics in regression_models
    regression_models, timings_2 = single_chunk_testing(file_handler = file_handler, 
        regression_models = regression_models,
        use_logicle_scaling=use_logicle_scaling, 
        normalization_method=normalization_method,  
        verbosity=VERBOSITY)

    # Carry out predictions on either the pooled events from each file or on a separate reference
    sub_p_adata, timings_3 = make_flow_regression_predictions(file_handler = file_handler, 
        regression_models = regression_models, 
        separate_backbone_reference = separate_backbone_reference,
        use_logicle_scaling=use_logicle_scaling, 
        normalization_method=normalization_method, 
        verbosity=VERBOSITY)

    if "batch" in sub_p_adata.obs.columns.values:
        sub_p_adata.obs["batch"] = sub_p_adata.obs["batch"].astype(str)

    if "name" in sub_p_adata.var.columns.values:
        sub_p_adata.var["name"] = sub_p_adata.var["name"].astype(str)

    # Downsample the final Infinity Flow object, if necessary
    n_events = sub_p_adata.obs.shape[0]
    if (n_final < n_events) and (n_final > 0):
        sub_p_adata = sub_p_adata[sub_p_adata.obs.sample(n_final, 
            random_state=RANDOM_STATE).index.values,:]

    # Carry out background correction, if necessary
    if file_handler.use_isotype_controls:
        background_corrected_data, background_corrected_var, timings_4 = \
            perform_background_correction(sub_p_adata = sub_p_adata,
                infinity_marker_annotation = infinity_marker_annotation, 
                file_handler = file_handler,
                cores_to_use = cores_to_use, 
                verbosity = VERBOSITY)
    else:
        timings_4 = {}

    # Use only the features from the backbone and prediction channels for
    # downstream analysis steps
    if file_handler.use_isotype_controls:
        infinity_markers = np.setdiff1d(infinity_marker_annotation.iloc[:,2].values,
            np.unique(infinity_marker_annotation.iloc[:,3].values))
    else:
        infinity_markers = infinity_marker_annotation.iloc[:,2].values

    features_to_use = np.union1d(backbone_annotation.iloc[:,0].values,
        infinity_markers)
    features_to_silence = np.setdiff1d(sub_p_adata.var.index.values,
        features_to_use)
    if len(features_to_silence) > 0:
        sub_p_adata = move_features_to_silent(sub_p_adata, features_to_silence)

    # Check make_feature_plots and add_umap for input validity
    if make_feature_plots:
        if not add_umap:
            printv(VERBOSITY, v1="WARNING! If make_feature_plots is set to true, \n"\
                "add_umap must also be set to True. Resetting add_umap to True...")
            add_umap = True

    # Perform PCA, if necessary
    if use_pca:
        sc.tl.pca(sub_p_adata)
        sub_p_adata.uns['pca_features'] = sub_p_adata.var.index.values
        make_pca_elbo_plot(sub_p_adata=sub_p_adata, output_paths=output_paths)

    # Find neighbors, if necessary (for umap and clustering)
    if add_umap or find_clusters:
        t_find_neighbors_start = time.time()
        if use_pca:
            printv(VERBOSITY, v1="Finding neighbors using PCA result...")
            sc.pp.neighbors(sub_p_adata, n_pcs=n_pc)
        else:
            printv(VERBOSITY, v1="Finding neighbors using raw data...")
            sc.pp.neighbors(sub_p_adata)
        
        t_find_neighbors_end = time.time()
        timings_neighbors = {\
            "find_neighbors": t_find_neighbors_end - t_find_neighbors_start}
    else:
        timings_neighbors = {}

    # Add umap, if necessary
    if add_umap:
        printv(VERBOSITY, v1="Adding umap to AnnData object...")
        t_add_umap_start = time.time()
        sc.tl.umap(sub_p_adata, random_state=RANDOM_STATE)
        sub_p_adata.obs["umap-x"] = sub_p_adata.obsm['X_umap'][:,0]
        sub_p_adata.obs["umap-y"] = sub_p_adata.obsm['X_umap'][:,1]
        t_add_umap_end = time.time()
        timings_umap = {\
            "add_umap": t_add_umap_end - t_add_umap_start}
    else:
        timings_umap = {}

    # Add clusters, if necessary
    if find_clusters:
        t_leiden_start = time.time()
        printv(VERBOSITY, v1="Finding clusters using leiden clustering...")
        sc.tl.leiden(sub_p_adata)
        t_leiden_end = time.time()
        timings_clustering = {\
            "leiden_clustering": t_leiden_end - t_leiden_start}
        groups_to_colors = assign_rainbow_colors_to_groups(\
            sub_p_adata.obs["leiden"].values)
        sub_p_adata.uns['groups_to_color'] = groups_to_colors
    else:
        timings_clustering = {}

    if find_clusters and add_umap:
        printv(VERBOSITY, prefix_debug=False, 
            v1="Plotting Leiden clusters over UMAP...")
        plot_leiden_clusters_over_umap(\
            sub_p_adata=sub_p_adata, 
            output_paths=output_paths, 
            verbosity=VERBOSITY)
        

    # Find markers for clusters
    if find_markers:
        printv(VERBOSITY, prefix_debug=False, 
            v1="Finding Markers for Leiden clusters in Infinity Flow object...")
        t_find_markers_start = time.time()
        markers_df, cell_assignments = find_markers_from_anndata(\
            sub_p_adata=sub_p_adata, 
            output_paths=output_paths, 
            groups_to_colors=sub_p_adata.uns['groups_to_color'], 
            verbosity=VERBOSITY)
        sub_p_adata.uns['markers_df'] = markers_df
        sub_p_adata.uns['cell_assignments'] = cell_assignments
        t_find_markers_end = time.time()
        timings_find_markers = {\
            "find_markers": t_find_markers_end - t_find_markers_start}
    else:
        timings_find_markers = {}

    # Output individual fcs files for each cluster
    # ADD LATER


    # Replace the silent features
    if len(features_to_silence) > 0:
        sub_p_adata = move_features_out_of_silent(sub_p_adata, 
            features_to_silence)

    # Save the umap figures
    if make_feature_plots:
        timings_6 = save_umap_figures_all_features(sub_p_adata, 
            background_corrected_data = background_corrected_data, 
            file_handler = file_handler, 
            output_paths = output_paths, 
            verbosity=VERBOSITY)
    else:
        timings_6 = {}

    # Save data to fcs file
    if file_handler.use_isotype_controls:
        timings_7 = save_fcs_flow_anndata(sub_p_adata = sub_p_adata, 
            background_corrected_data = background_corrected_data, 
            background_corrected_var = background_corrected_var, 
            file_handler = file_handler, 
            output_paths = output_paths, 
            verbosity=VERBOSITY)
    else:
        timings_7 = save_fcs_flow_anndata(sub_p_adata = sub_p_adata, 
            background_corrected_data = None, 
            background_corrected_var = None, 
            file_handler = file_handler, 
            output_paths = output_paths,
            add_umap = add_umap, 
            verbosity=VERBOSITY)

    # Save to h5ad (AnnData object) if desired
    if save_h5ad:
        printv(VERBOSITY, v1 = "Saving Infinity Flow object as h5ad file...")
        sub_p_adata.write(os.path.join(output_paths["output_regression_path"],
            "infinity_flow_object_logicle_normalized.h5ad"))

    # Save to feather file if desired
    if save_feather:
        printv(VERBOSITY, v1 = "Saving Infinity Flow object as feather file...")
        anndata_to_df(sub_p_adata, use_raw_feature_names=False).reset_index().\
            to_feather(os.path.join(output_paths["output_regression_path"],
                "infinity_flow_object_logicle_normalized.feather"))

    # Save the file_handler object if desired
    if save_file_handler:
        with open(os.path.join(output_paths['qc'], "file_handler.pickle"), "wb") as tmp_file:
            pickle.dump(file_handler, tmp_file)

    # Save the regression_models object if desired
    if save_regression_models:
        with open(os.path.join(output_paths['qc'], "regression_models.pickle"), "wb") as tmp_file:
            pickle.dump(regression_models, tmp_file)

    t_end_base_inflow_pipeline = time.time()

    print("Done.")
    print("Base InfinityFlow Pipeline took {:.2f} seconds for input "\
        "dataset.".format(t_end_base_inflow_pipeline - t_start_base_inflow_pipeline))

    timing_series = pd.Series({**timings_1,
                                **timings_2,
                                **timings_3,
                                **timings_4,
                                **timings_6,
                                **timings_7,
                                **timings_neighbors,
                                **timings_umap,
                                **timings_clustering,
                                **timings_find_markers})
                                
    timing_series.to_csv(os.path.join(output_paths["qc"],
        "InfinityFlow_Process_Timings.csv"), header=False, index=True)



if __name__ == "__main__":
    main()

