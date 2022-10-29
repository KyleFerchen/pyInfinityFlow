# Command Line Tools

## pyInfinityFlow

### Usage

#### Simple

#### All Arguments

Here is an example of calling the command line program to run the pipeline specifying every argument. This is intended to be useful for quick copying and pasting of all command arguments to later remove selected and update values of modified arguments. The settings of the arguments here attempt to replicate how the original author's processed their published[<sup>1</sup>](https://www.science.org/doi/full/10.1126/sciadv.abg0505) mouse lung [dataset](https://flowrepository.org/id/FR-FCM-Z2LP) using the BioLegend Murine LEGENDScreen kit (Eg. use 50% of all measured events from InfinityMarker files and pool 10,000 cells from each InfinityMarker final into the final InfinityFlow object):

```console
pyInfinityFlow --data_dir "pyInfinityFlow/data/fcs_files/" \
                --outdir "pyInfinityFlow/output_dir/" \
                --backbone_annotation "pyInfinityFlow/data/small_test_dataset_backbone_anno.csv" \
                --infinity_marker_annotation "pyInfinityFlow/data/small_test_dataset_infinity_marker_anno.csv" \
                --random_state 7 \
                --use_logicle_scaling True \
                --normalization_method "zscore" \
                --n_events_train 0 \
                --n_events_validate 0 \
                --ratio_for_validation 0.5 \
                --separate_backbone_reference None \
                --n_events_combine 10000 \
                --n_final 0 \
                --add_umap True \
                --find_clusters True \
                --find_markers True \
                --make_feature_plots True \
                --use_pca True \
                --n_pc 15 \
                --n_pc_plot_qc 50 \
                --save_h5ad True \
                --save_feather True \
                --save_file_handler True \
                --save_regression_models True \
                --verbosity 1 \
                --n_cores 12
```

### Required Arguments

``--data_dir`` ***(str)***: Directory in which the .fcs files are contained. Each InfinityMarker must be associated with an .fcs file in this directory. Each .fcs file must have the channel names present in the backbone_annotation file. It is expected that the values of the .fcs data are already compensated and don't require adjustment with spillover. For example, the "export compensated values" feature from FlowJo could be used to export compensated Flow Cytometry data.

<hr />

``--outdir`` ***(str)***: Directory in which to save the outputs.

<hr />

``--backbone_annotation`` ***(str)***: Path to the backbone annotation file. It can be either a CSV (.csv) or TSV (.tsv or .txt) file with a header that annotates the column names. Each subsequent line annotates a backbone channel for the regression models, which will be used as predictors for the targets (InfinityMarkers). For each backbone channel, the following columns are annotated:

1. **Reference_Backbone:** the name of the backbone channel as it appears in the ``--separate_backbone_reference`` file (if a separate reference is used) or the InfinityMarker files if ``--n_events_combine`` is used to pool events from InfinityMarkers into a final InfinityFlow object
2. **Query_Backbone:** the name of the backbone channel as it is written in the InfinityMarker files (if the channel name of backbone parameters is different between InfinityMarker files, the pipeline must be run multiple times for each backbone channel layout)
3. **Final_Name:** the desired name parameter in the final InfinityFlow object

*It is recommended to build this file in Excel and export as a .csv file. Here is an example of the structure of a "backbone_annotation" file:*

| Reference_Backbone | Query_Backbone | Final_Name |
| ------------------ | -------------- | ---------- |
| <font size="1">FJComp-APC-A</font> | <font size="1">FJComp-APC-A</font> | <font size="1">CD69-CD301b</font> |
| <font size="1">FJComp-AlexaFluor700-A</font> | <font size="1">FJComp-AlexaFluor700-A</font> | <font size="1">MHCII</font> |
| <font size="1">FJComp-BUV395-A</font> | <font size="1">FJComp-BUV395-A</font> | <font size="1">CD4</font> |
| <font size="1">FJComp-BUV737-A</font> | <font size="1">FJComp-BUV737-A</font> | <font size="1">CD44</font> |
| <font size="1">FJComp-BV421-A</font> | <font size="1">FJComp-BV421-A</font> | <font size="1">CD8</font> |
| <font size="1">FJComp-BV510-A</font> | <font size="1">FJComp-BV510-A</font> | <font size="1">CD11c</font> |
| <font size="1">FJComp-BV605-A</font> | <font size="1">FJComp-BV605-A</font> | <font size="1">CD11b</font> |
| <font size="1">FJComp-BV650-A</font> | <font size="1">FJComp-BV650-A</font> | <font size="1">F480</font> |
| <font size="1">FJComp-BV711-A</font> | <font size="1">FJComp-BV711-A</font> | <font size="1">Ly6C</font> |
| <font size="1">FJComp-BV786-A</font> | <font size="1">FJComp-BV786-A</font> | <font size="1">Lineage</font> |
| <font size="1">FJComp-GFP-A</font> | <font size="1">FJComp-GFP-A</font> | <font size="1">CD45a488</font> |
| <font size="1">FJComp-PE-Cy7(yg)-A</font> | <font size="1">FJComp-PE-Cy7(yg)-A</font> | <font size="1">CD24</font> |
| <font size="1">FJComp-PerCP-Cy5-5-A</font> | <font size="1">FJComp-PerCP-Cy5-5-A</font> | <font size="1">CD103</font> |



<hr />

``--infinity_marker_annotation`` ***(str)***: The path to the file which annotates each InfinityMarker. It can be either a CSV (.csv) or TSV (.tsv or .txt) file with a header that annotates the column names. Each subsequent line of the file annotates an InfinityMarker with the following columns:

1. **File:** the name of the file as it is saved in the ``--data_dir``
2. **Channel:** the name of the channel, as it is written in the .fcs File
3. **Name:** the desired name to give to the channel in the final InfinityFlow object (should be unique)
4. **Isotype:** ***(OPTIONAL)*** the InfinityMarker that serves as the isotype for the given InfinityMarker (must match one of the **other** Name values in the third column)

*It is recommended to build this file in Excel and export as a .csv file. Here is an example of the structure of an "infinity_marker_annotation" file:*

| File | Channel | Name | Isotype<br />*(Optional)* |
| ---- | ------- | ---- | ------- |
| <font size="1">backbone_Plate1_Specimen_001_A11_A11_011_target_CD11b.fcs</font> | <font size="1">FJComp-PE(yg)-A</font> | <font size="1">CD11b</font> | <font size="1">rIgG2b</font> |
| <font size="1">backbone_Plate1_Specimen_001_A12_A12_012_target_CD11c.fcs</font> | <font size="1">FJComp-PE(yg)-A</font> | <font size="1">CD11c</font> | <font size="1">AHIgG</font> |
| <font size="1">backbone_Plate1_Specimen_001_B11_B11_023_target_CD27.fcs</font> | <font size="1">FJComp-PE(yg)-A</font> | <font size="1">CD27</font> | <font size="1">AHIgG</font> |
| <font size="1">backbone_Plate1_Specimen_001_B2_B02_014_target_CD16-32.fcs</font> | <font size="1">FJComp-PE(yg)-A</font> | <font size="1">CD16-32</font> | <font size="1">rIgG2a</font> |
| <font size="1">backbone_Plate1_Specimen_001_D2_D02_038_target_CD45R-B220.fcs</font> | <font size="1">FJComp-PE(yg)-A</font> | <font size="1">CD45R-B220</font> | <font size="1">rIgG2a</font> |
| <font size="1">backbone_Plate1_Specimen_001_E12_E12_060_target_CD71.fcs</font> | <font size="1">FJComp-PE(yg)-A</font> | <font size="1">CD71</font> | <font size="1">rIgG2a</font> |
| <font size="1">backbone_Plate1_Specimen_001_E2_E02_050_target_CD55.fcs</font> | <font size="1">FJComp-PE(yg)-A</font> | <font size="1">CD55</font> | <font size="1">AHIgG</font> |
| <font size="1">backbone_Plate1_Specimen_001_G10_G10_082_target_CD117 (c-kit).fcs</font> | <font size="1">FJComp-PE(yg)-A</font> | <font size="1">CD117</font> | <font size="1">rIgG2b</font> |
| <font size="1">backbone_Plate3_Specimen_001_B3_B03_015_target_Ly-6C.fcs</font> | <font size="1">FJComp-PE(yg)-A</font> | <font size="1">Ly-6C</font> | <font size="1">rIgG2c</font> |
| <font size="1">backbone_Plate3_Specimen_001_B5_B05_017_target_Ly-6G.fcs</font> | <font size="1">FJComp-PE(yg)-A</font> | <font size="1">Ly-6G</font> | <font size="1">rIgG2a</font> |
| <font size="1">backbone_Plate3_Specimen_001_F11_F11_071_target_Isotype_rIgG2a.fcs</font> | <font size="1">FJComp-PE(yg)-A</font> | <font size="1">rIgG2a</font> | <font size="1">rIgG2a</font> |
| <font size="1">backbone_Plate3_Specimen_001_F12_F12_072_target_Isotype_rIgG2b.fcs</font> | <font size="1">FJComp-PE(yg)-A</font> | <font size="1">rIgG2b</font> | <font size="1">rIgG2b</font> |
| <font size="1">backbone_Plate3_Specimen_001_G1_G01_073_target_Isotype_rIgG2c.fcs</font> | <font size="1">FJComp-PE(yg)-A</font> | <font size="1">rIgG2c</font> | <font size="1">rIgG2c</font> |
| <font size="1">backbone_Plate3_Specimen_001_F4_F04_064_target_Isotype_AHIgG.fcs</font> | <font size="1">FJComp-PE(yg)-A</font> | <font size="1">AHIgG</font> | <font size="1">AHIgG</font> |


### Optional Arguments

``--random_state`` ***(int|None)*** **(Default=``None``):** Integer to specify the random_state of sampling, regression, and UMAP to make results more reproducible.

<hr />

``--use_logicle_scaling`` ***(True|False)*** **(Default=``True``)**: Whether or not to apply logicle scaling to features that are typically fluorescence channels in Flow Cytometry and not common linear features (Eg. FSC-A, SSC-A, ...)

<hr />

``--normalization_method`` ***("zscore"|None)*** **(Default=``"zscore"``)**: Method used for normalizing backbone feature values before regression in an effort to reduce sample to sample batch effects.

<hr />

``--n_events_train`` ***(int)*** **(Default=``0``)**: Integer to specify the number events to use for training. ``0`` is a special case in which all events from the file will be used, in which case, ``--n_events_validate`` must also be set to ``0`` and ``--ratio_for_validation`` must be greater than ``0`` and less than ``1.0``. The sum of ``--n_events_train`` and ``--n_events_validate`` must not exceed the number of events in any of the InfinityMarker .fcs files annotated in the ``--infinity_marker_annotation`` file.

<hr />

``--n_events_validate`` ***(int)*** **(Default=``0``)**: Integer to specify the number events to use for validation. ``0`` is a special case in which all events from the file will be used, in which case, ``--n_events_train`` must also be set to ``0`` and ``--ratio_for_validation`` must be greater than ``0`` and less than ``1.0``. The sum of ``--n_events_train`` and ``--n_events_validate`` must not exceed the number of events in any of the InfinityMarker .fcs files annotated in the ``--infinity_marker_annotation`` file.

<hr />

``--ratio_for_validation`` ***(0 < float < 1)*** **(Default=``0.2``)**: If ``--n_events_train`` and ``--n_events_validate`` are both 0, then all of the events from the fcs file will be used and this argument will will specify what percentage of the dataset should be used for validation and the remainder will be used for training.

<hr />

``--separate_backbone_reference`` ***(str | None)*** **(Default=``None``)**: The .fcs file passed as a file path string. This can be used as an alternative to ``--n_events_combine``. The regression will then be applied to the events in this file. Each of the Infinity Markers specified in the infinity_marker_annotation file and the original channel values for this ``--separate_backbone_reference`` will be in the final output.

<hr />

``--n_events_combine`` ***(int|None)*** **(Default=``None``)**: As an alternative to using a separate, external, reference .fcs file, the ``--n_events_combine`` argument can be used to pool events from the InfinityMarker input .fcs files specified in the ``--infinity_marker_annotation`` file to sample events for the final InfinityFlow object. The ``--random_state`` argument will set the seed for this sampling. The resulting InfinityFlow object will be made up of an even sample of ``--n_events_combine`` from each unique InfinityMarker file. ``0`` is a special case in which all events from the InfinityMarker files will be pooled together.

<hr />

``--n_final`` ***(int)*** **(Default=``0``)**: Specifies the number of events to include in the final InfinityFlow object. This will either sample from the ``--separate_backbone_reference`` file (if not set to None), in which case the value needs to be <= the number of events in that file; or from the pooled cells specified by ``--n_events_combine``, in which case it needs to be less than the sum of the pooled events.

<hr />

``--add_umap`` ***(True|False)*** **(Default=``False``)**: Boolean to specify if UMAP dimensionality reduction should be carried out on the final InfinityFlow object to reduce to 2 dimensions for visualization.

<hr />

``--find_clusters`` ***(True|False)*** **(Default=``False``)**: Boolean to specify clustering should be done using Leiden clustering implemented through the Scanpy package.

<hr />

``--find_markers`` ***(True|False)*** **(Default=``False``)**: Boolean to specify if MarkerFinder should be applied to find optimal markers for clusters. ``--find_clusters`` must be set to ``True`` to use this feature.

<hr />

``--make_feature_plots`` ***(True|False)*** **(Default=``False``)**: Boolean to specify if each feature in the final InfinityFlow object should be plotted over the 2D UMAP embedding.

<hr />

``--use_pca`` ***(True|False)*** **(Default=``True``)**: Boolean to specify if principal component ananlysis should be used to reduce the feature space prior to UMAP and clustering. This is suggested to save computation time.

<hr />

``--n_pc`` ***(int)*** **(Default=``15``)**: Integer to specify the number principal components to use for UMAP and clustering. It is recommended to look at the PC-elbo curve in the outputs to refine the optimal number of principal components to use. This value must be less than the total number of features in the final InfinityFlow object (InfinityMarkers + backbone channels) defined in the ``--infinity_marker_annotation`` file and ``--backbone_annotation`` file.

<hr />

``--n_pc_plot_qc`` ***(int)*** **(Default=``50``)**: Integer to specify the number principal components to plot in the elbo curve. Helpful for estimating the number of principal components to use downstream.

<hr />

``--save_h5ad`` ***(True|False)*** **(Default=``False``)**: Boolean to specify if the final InfinityFlow object should be saved as an h5ad file. Useful for quick loading of the data into a Python anndata.AnnData object for downstream analyses with Scanpy.

<hr />

``--save_feather`` ***(True|False)*** **(Default=``False``)**: Boolean to specify if the final InfinityFlow object should be saved as a DataFrame in a feather file. Useful for quick loading of the data into a Python Pandas DataFrame.

<hr />

``--save_file_handler`` ***(True|False)*** **(Default=``False``)**: Boolean to specify whether or not to save the intermediate pyInfinityFlow.file_handler object, which stores data on how each of the InfinityMarker .fcs files were processed.

<hr />

``--save_regression_models`` ***(True|False)*** **(Default=``False``)**: Boolean to specify whether or not to save the intermediate pyInfinityFlow.regression_models object, which stores the regression models and validation metrics for each InfinityMarker feature.

<hr />

``--verbosity`` ***(0|1|2|3)*** **(Default=``1``)**: The level of verbosity with which to write to std-out. ``0`` = no print statements, to ``3`` = all debug print statements.

<hr />

``--n_cores`` ***(int)*** **(Default=``1``)**: The number of cores to use, which can increase the speed of regression fitting, UMAP dimensionality reduction, and Leiden clustering.

### Outputs

The outputs of the pipeline are written to the path specified by the ``--outdir`` argument in the following tree structure:

---

---

## pyInfinityFlow-list_channels

This command line tool will list out the existing channels in a given FCS file.

This is useful for building the InfinityMarker and Backbone annotation files, so that you can use the correct format and spelling for the name of a given channel.

```console
pyInfinityFlow-list_channels --fcs_file "" \
    --add_user_defined_names
```
