import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.colors import to_hex
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
import seaborn as sns

from pyInfinityFlow.Transformations import scale_feature
from pyInfinityFlow.Debugging_Utilities import printv


# matplotlib.use('Qt5Agg') # For interactive plotting in Linux Ubuntu

# Define a colormap to use for marker expression heatmap
N = 256
vals = np.ones((N*2, 4))
vals[:N, 0] = np.linspace(15/256, 0, N)
vals[:N, 1] = np.linspace(255/256, 0, N)
vals[:N, 2] = np.linspace(255/256, 0, N)
vals[N:, 0] = np.linspace(0, 255/256, N)
vals[N:, 1] = np.linspace(0, 243/256, N)
vals[N:, 2] = np.linspace(0, 15/256, N)
blue_black_yellow_cmap = ListedColormap(vals)

def assign_rainbow_colors_to_groups(groups):
    """ Creates a dictionary of cluster names to hexadecimal color strings

    This function takes a list of groups and assigns each unique item in the 
    groups a color (using the matplotlib.cm.rainbow color-map) as a hexadecimal 
    string value. This is useful for storing a single color scheme for clusters 
    to be used with downstream visualizations.

    Arguments
    ---------
    groups : numpy.Array[str]
        List of cluster names. (Required)
    
    Returns
    -------
    dict {str:str}
        Dictionary of cluster-names to assigned colors (hexadecimal value)

    """
    unique_groups = np.unique(groups)
    groups_to_num = pd.Series(list(range(len(unique_groups))),
        index=unique_groups)
    n = len(unique_groups)
    groups_to_color = pd.Series([to_hex(cm.rainbow(item/n)) for item in \
        groups_to_num.values], index=groups_to_num.index.values).to_dict()
    return(groups_to_color)

    

def plot_feature_over_2d_umap_from_anndata_and_save_fig(input_anndata, feature, file_path):
    try:
        x = input_anndata.obs["umap-x"].values
        y = input_anndata.obs["umap-y"].values
    except Exception as e:
        print("ERROR! Couldn't find umap coordinates in input_anndata object.")
    plot_value = scale_feature(input_anndata[:,feature].X.toarray().reshape(-1), min_threshold_percentile=1, max_threshold_percentile=99)

    divnorm=matplotlib.colors.TwoSlopeNorm(vmin=0.2, vcenter=0.5, vmax=0.8)
    plot_value=divnorm(plot_value)
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(x, y, c=plot_value, s=1, cmap="jet", alpha=0.5)
    ax.grid(False)
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=divnorm, cmap="jet"), ax=ax)
    cbar.set_label(feature)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.set_figwidth(12)
    fig.set_figheight(12)
    plt.title(f'{feature} : {input_anndata.var.loc[feature, "name"]}')
    plt.savefig(file_path)
    return()


def plot_feature_over_x_y_coordinates_and_save_fig(feature_vector, x, y, feature_name, file_path):
    """ Plots a 2D-scatter plot of numeric vector over x and y coordinates

    This function takes a feature_vector, x and y coordinates, a feature_name, 
    and a file_path and plots a scatterplot of all points, coloring the points 
    using the "jet" colormap in matplotlib following the feature_vector scale. 
    
    Warning
    -------
    It is expected that feature_vector, x, and y correspond to the same events, 
    in the same order.

    Note
    ----
    The colormap will start at the 20th percentile (~blue) and end at the 
    80th percentile (~red) of the feature vector.

    Arguments
    ---------
    feature_vector : numpy.Array[numeric]
        Numeric values to map the 'jet' colormap onto in the scatter plot. \
        (Required)
    x : numpy.Array[numeric]
        Numeric values for the x-coordinate of the scatter plot (Required)
    y : numpy.Array[numeric]
        Numeric values for the y-coordinate of the scatter plot (Required)
    feature_name : str
        Label to give the colorbar and plt.title of the scatter plot. \
        (Required)
    file_path : str
        The path to save the figure. (Required)
    
    Returns
    -------
    None
        Saves the scatterplot to the file specified by file_path

    """
    # matplotlib.use('Agg')
    plot_value = scale_feature(feature_vector, 
        min_threshold_percentile=1, 
        max_threshold_percentile=99)
    divnorm=matplotlib.colors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    plot_value=divnorm(plot_value)
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(x, y, c=plot_value, s=1, cmap="jet", alpha=0.5)
    ax.grid(False)
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=divnorm, cmap="jet"), ax=ax)
    cbar.set_label(feature_name)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.set_figwidth(12)
    fig.set_figheight(12)
    plt.title(feature_name)
    plt.savefig(file_path)
    return()


def plot_feature_over_2d_umap_from_df_and_save_fig(input_umap, input_df, feature, feature_name, file_path):
    try:
        x = input_umap[:,0]
        y = input_umap[:,1]
    except Exception as e:
        print("ERROR! UMAP coordinates not in np.array format with shape n x 2.")
    plot_value = scale_feature(input_df.loc[:,feature].values, min_threshold_percentile=1, max_threshold_percentile=99)
    divnorm=matplotlib.colors.TwoSlopeNorm(vmin=0.2, vcenter=0.5, vmax=0.8)
    plot_value=divnorm(plot_value)
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(x, y, c=plot_value, s=1, cmap="jet", alpha=0.5)
    ax.grid(False)
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=divnorm, cmap="jet"), ax=ax)
    cbar.set_label(feature)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.set_figwidth(12)
    fig.set_figheight(12)
    plt.title(f'{feature} : {feature_name}')
    plt.savefig(file_path)
    return()


"""
Method for plotting isotype background correction linear models
"""
def plot_lm_model_results(input_x, input_y, slope, intercept, plot_name, x_label, y_label):
    plt.close("all")
    fig, ax = plt.subplots()
    ax.scatter(input_x, input_y, s=0.25, alpha=0.5, c="#A89EA8")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    current_xlim = ax.get_xlim()
    current_ylim = ax.get_ylim()
    line_x = np.array(current_xlim)
    line_y = intercept + slope * line_x
    plt.plot(line_x, line_y, '--', c="#DB382D")
    adj_x = 0.1 * (current_xlim[1] - current_xlim[0])
    adj_y = 0.1 * (current_ylim[1] - current_ylim[0])
    new_xlim = (current_xlim[0] - adj_x, current_xlim[1] + adj_x)
    new_ylim = (current_ylim[0] - adj_y, current_ylim[1] + adj_y)
    ax.set_xlim(new_xlim)
    ax.set_ylim(new_ylim)
    ax.text(new_xlim[1], 
            current_ylim[0],
            "y = {:.2f} + {:.2f} * x + \u03B5".format(intercept, slope),
            c="#F5C290", ha="right", va="top")
    plt.title(plot_name, color="#8DAC50")
    ax.set_ylabel(y_label, c="#8DAC50")
    ax.set_xlabel(x_label, c="#8DAC50")
    ax.tick_params(color='#613B6F', labelcolor='#613B6F')
    plt.show()
    return(current_xlim)

# # Example utilizing the plot_lm_model_results function
# test = plot_lm_model_results(input_x=tmp_x, 
#                         input_y=tmp_y, 
#                         slope=slope, 
#                         intercept=intercept,
#                         plot_name="{} Background Correction".format(infinity_marker),
#                         x_label="Isotype {} Expression".format(iso),
#                         y_label="Infinity Marker {} Expression".format(infinity_marker))



def plot_markers_df(input_df, ordered_markers_df, ordered_cells_df, 
        groups_to_colors, path_to_save_figure):
    """ Plots a heatmap of the MarkerFinder results

    This function takes a pandas.DataFrame of values, a markers_df and 
    cell_assignments from pyInfinityFlow.InfinityFlow_Utilities.
    find_markers_from_anndata to plot a heatmap of the markers.

    Note
    ----
    This function expects pyInfinityFlow.InfinityFlow_Utilities.
    find_markers_from_anndata to have already been run.
        
    Arguments
    ---------
    input_df : pandas.DataFrame
        Data to plot. The columns must intersect with features in the \
        ordered_markers_df and the rows must intersect with the cells in \
        (Required)
    ordered_markers_df : pandas.DataFrame
        The markers_df output from pyInfinityFlow.InfinityFlow_Utilities.\
        find_markers_from_anndata (Required)
    ordered_cells_df : pandas.DataFrame
        The cell_assignments output from pyInfinityFlow.InfinityFlow_Utilities.\
        find_markers_from_anndata (Required)
    groups_to_colors : dict {str:str}
        Dictionary of cluster-names to assigned colors (hexadecimal value) \
        The pyInfinityFlow.Plotting_Utilities.assign_rainbow_colors_to_groups \
        can be used to generate this dictionary from a list of clusters. \
        (Required)
    path_to_save_figure : str
        The path to save the figure. (Required)
    
    Returns
    -------
    None
        Saves the heatmap to the file specified by path_to_save_figure

    """
    try:
        # Filter the input data matrix to the markers and cells of interest
        input_df = input_df.loc[ordered_cells_df["cell"].values, 
            ordered_markers_df["marker"].values].T
        # Map the order of the groups_to_colors to make a cmap
        groups_to_order = pd.Series(list(range(len(groups_to_colors))),
            index=groups_to_colors.index.values)
        # Build top heatmap to label clusters
        cell_labels_df = pd.DataFrame({"cluster": [groups_to_order[item] for \
                item in ordered_cells_df["top_cluster"].values]}, 
            index=ordered_cells_df["cell"].values).T
        # Build df to add cluster label ticks
        label_to_position = pd.pivot_table(pd.DataFrame({\
                "cluster": ordered_cells_df["top_cluster"].values,
                "position": list(range(ordered_cells_df.shape[0]))}), 
            index="cluster", 
            values="position", 
            aggfunc=np.mean)["position"].sort_values()
        # Z-score normalize the expression matrix
        z_df = input_df.apply(lambda x: pd.Series(zscore(x.values), 
            index=x.index.values), axis=1)
        # Build the heatmap
        plt.close("all")
        fig = plt.figure(constrained_layout=True, figsize=(16,8))
        ax = fig.add_gridspec(20, 1)
        ax1 = fig.add_subplot(ax[:1, 0])
        ax2 = fig.add_subplot(ax[1:, 0])
        heat1 = sns.heatmap(cell_labels_df,
            yticklabels=False,
            xticklabels=False,
            cmap=sns.color_palette(groups_to_colors.values),
            cbar=False,
            ax=ax1)
        ax1.set_xticks(label_to_position.values)
        ax1.set_xticklabels(label_to_position.index.values)
        ax1.xaxis.tick_top()
        heat2 = sns.heatmap(z_df, 
            vmin=-3, 
            vmax=3, 
            cmap=blue_black_yellow_cmap,
            xticklabels=False,
            yticklabels=True,
            cbar=True,
            cbar_kws={"shrink": 0.5},
            ax=ax2)
        ax2.collections[0].colorbar.set_label("Z-Score Normalized Expression")
        fig.suptitle('Sampled Cluster Markers', fontsize=16)
        plt.savefig(path_to_save_figure)
    except Exception as e:
        print(str(e))
        print("Warning! Failed to run plot_markers_df. See above Exception.")



def plot_leiden_clusters_over_umap(sub_p_adata, output_paths, verbosity):
    """ Plots a 2D-UMAP colored by the values in the "leiden" field

    This function takes a pandas.DataFrame of values, a markers_df and 
    cell_assignments from pyInfinityFlow.InfinityFlow_Utilities.
    find_markers_from_anndata to plot a heatmap of the markers.

    Note
    ----
    It is expected that scanpy.pp.neighbors, scanpy.tl.umap, and scanpy.tl.
    leiden have been run on sub_p_adata. Also the x,y coordinates of the UMAP 
    must have been added to the sub_p_adata.obs pandas.DataFrame.
        
    Arguments
    ---------
    sub_p_adata : anndata.AnnData
        pyInfinityFlow formatted AnnData object. It is expected that the object 
        have the following attributes present:
            - sub_p_adata.obs['umap-x'] : the x-coordinates of the UMAP plot are \
            required to be in the sub_p_adata.obs pandas.DataFrame
            - sub_p_adata.obs['umap-y'] : the y-coordinates of the UMAP plot are \
            required to be in the sub_p_adata.obs pandas.DataFrame
            - sub_p_adata.obs['leiden'] : leiden cluster assignments are \
            required to be in the sub_p_adata.obs pandas.DataFrame
            - sub_p_adata.uns['groups_to_color'] : (dict{str:str}) Dictionary of \
            cluster-names to assigned colors (hexadecimal value) (Required)
    output_paths: dict
        The output_paths dictionary created by the pyInfinityFlow.\
        InfinityFlow_Utilities.setup_output_directories function (Required)
    verbosity: int (0|1|2|3) 
        Specifies to what verbosity level the function will output progress and \
        debugging statements. (Default=0)
    
    Returns
    -------
    None
        Saves the scatterplot to the file specified by output_paths["clustering"]

    """
    try:
        plt.close("all")
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.grid(False)
        # Make UMAP scatter plot
        ax.scatter(sub_p_adata.obs['umap-x'].values, 
                    sub_p_adata.obs['umap-y'].values,
                    c=[sub_p_adata.uns['groups_to_color'][item] for item in \
                        sub_p_adata.obs["leiden"].values],
                    s=0.25,
                    alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title("UMAP: Leiden Clustering")
        # Get the centroid positions of clusters in UMAP
        cluster_centroids = pd.pivot_table(\
            sub_p_adata.obs, 
            values=["umap-x", "umap-y"], 
            index="leiden", 
            aggfunc=np.mean)
        # Add labels for Leiden clusters
        for c_name, c_pos in cluster_centroids.iterrows():
            ax.add_patch(plt.Circle(\
                (c_pos["umap-x"],c_pos["umap-y"]),
                radius=0.5,
                facecolor="#ffffff"))
            ax.text(c_pos["umap-x"],c_pos["umap-y"], c_name, ha="center", va="center")
        plt.savefig(os.path.join(output_paths["clustering"], 
            "Leiden_Clusters_over_UMAP.png"))
    except Exception as e:
        printv(verbosity, v3=str(e))
        print("Warning! Failed to plot Leiden clusters.")