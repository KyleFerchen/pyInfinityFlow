# API
The pyInfinityFlow API is designed to give the user more control over what parameters are used in the InfinityFlow pipeline. It also allows for any FCS file to be processed with any step of the pipeline.

## InfinityFlow_Utilities

### InfinityFlow_Utilities: Classes
```{eval-rst}
.. autoclass:: pyInfinityFlow.InfinityFlow_Utilities.InfinityFlowFileHandler
    :members: add_handle
```

```{eval-rst}
.. autoclass:: pyInfinityFlow.InfinityFlow_Utilities.CombinedRegressionModels
    :members: init
```

---

### InfinityFlow_Utilities: General Tools
```{eval-rst}
.. autofunction:: pyInfinityFlow.InfinityFlow_Utilities.read_annotation_table
```

```{eval-rst}
.. autofunction:: pyInfinityFlow.InfinityFlow_Utilities.anndata_to_df
```

```{eval-rst}
.. autofunction:: pyInfinityFlow.InfinityFlow_Utilities.marker_finder
```

```{eval-rst}
.. autofunction:: pyInfinityFlow.InfinityFlow_Utilities.read_fcs_into_anndata
```

```{eval-rst}
.. autofunction:: pyInfinityFlow.InfinityFlow_Utilities.write_anndata_to_fcs
```

```{eval-rst}
.. autofunction:: pyInfinityFlow.InfinityFlow_Utilities.apply_logicle_to_anndata
```

```{eval-rst}
.. autofunction:: pyInfinityFlow.InfinityFlow_Utilities.apply_inverse_logicle_to_anndata
```

```{eval-rst}
.. autofunction:: pyInfinityFlow.InfinityFlow_Utilities.move_features_to_silent
```

```{eval-rst}
.. autofunction:: pyInfinityFlow.InfinityFlow_Utilities.move_features_out_of_silent
```

```{eval-rst}
.. autofunction:: pyInfinityFlow.InfinityFlow_Utilities.make_pca_elbo_plot
```

---


### InfinityFlow_Utilities: Analysis Pipeline Functions
```{eval-rst}
.. autofunction:: pyInfinityFlow.InfinityFlow_Utilities.check_infinity_flow_annotation_dataframes
```

```{eval-rst}
.. autofunction:: pyInfinityFlow.InfinityFlow_Utilities.setup_output_directories
```

```{eval-rst}
.. autofunction:: pyInfinityFlow.InfinityFlow_Utilities.single_chunk_training
```

```{eval-rst}
.. autofunction:: pyInfinityFlow.InfinityFlow_Utilities.single_chunk_testing
```

```{eval-rst}
.. autofunction:: pyInfinityFlow.InfinityFlow_Utilities.make_flow_regression_predictions
```

```{eval-rst}
.. autofunction:: pyInfinityFlow.InfinityFlow_Utilities.perform_background_correction
```

```{eval-rst}
.. autofunction:: pyInfinityFlow.InfinityFlow_Utilities.find_markers_from_anndata
```

```{eval-rst}
.. autofunction:: pyInfinityFlow.InfinityFlow_Utilities.save_umap_figures_all_features
```

```{eval-rst}
.. autofunction:: pyInfinityFlow.InfinityFlow_Utilities.save_fcs_flow_anndata
```

---

## Transformations

```{eval-rst}
.. autofunction:: pyInfinityFlow.Transformations.apply_logicle
```

```{eval-rst}
.. autofunction:: pyInfinityFlow.Transformations.apply_inverse_logicle
```

```{eval-rst}
.. autofunction:: pyInfinityFlow.Transformations.scale_feature
```

---

## fcs_io
### FCSFileObject Class
```{eval-rst}
.. autoclass:: pyInfinityFlow.fcs_io.FCSFileObject
    :members: load_data_from_pd_df,to_fcs,read_fcs
```

### fcs_io: Functions

```{eval-rst}
.. autofunction:: pyInfinityFlow.fcs_io.list_fcs_channels
```

---

## Plotting_Utilities

```{eval-rst}
.. autofunction:: pyInfinityFlow.Plotting_Utilities.assign_rainbow_colors_to_groups
```

```{eval-rst}
.. autofunction:: pyInfinityFlow.Plotting_Utilities.plot_feature_over_x_y_coordinates_and_save_fig
```

```{eval-rst}
.. autofunction:: pyInfinityFlow.Plotting_Utilities.plot_markers_df
```

```{eval-rst}
.. autofunction:: pyInfinityFlow.Plotting_Utilities.plot_leiden_clusters_over_umap
```
