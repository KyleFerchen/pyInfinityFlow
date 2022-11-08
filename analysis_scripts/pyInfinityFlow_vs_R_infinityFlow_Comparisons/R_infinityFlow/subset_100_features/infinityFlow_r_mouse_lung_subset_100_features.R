library(flowCore)
library(stats)
library(grDevices)
library(utils)
library(graphics)
library(pbapply)
library(matlab)
library(png)
library(raster)
library(grid)
library(uwot)
library(gtools)
library(Biobase)
library(generics)
library(parallel)
library(methods)
library(xgboost)

path_wd <- "/data/salomonis2/LabFiles/Kyle/Development/infinityFlow/R/test_scripts/mouse_lung_dataset_separate_directories/subset_100_features"
prefix_input <- "subset_100_features"
iso_anno <- read.csv("/data/salomonis2/LabFiles/Kyle/Development/pyInfinityFlow/data/mouse_lung_dataset_annotations/infinitymarker_annotation_100_feature_subset.csv")


setwd(path_wd)

source("/data/salomonis2/LabFiles/Kyle/Development/infinityFlow/R/mod_infinityFlow_src/00_master.R")
source("/data/salomonis2/LabFiles/Kyle/Development/infinityFlow/R/mod_infinityFlow_src/01_infinity_flow_fcs_parsing.R")
source("/data/salomonis2/LabFiles/Kyle/Development/infinityFlow/R/mod_infinityFlow_src/02_logicle_transforms.R")
source("/data/salomonis2/LabFiles/Kyle/Development/infinityFlow/R/mod_infinityFlow_src/03_fcs_scaling.R")
source("/data/salomonis2/LabFiles/Kyle/Development/infinityFlow/R/mod_infinityFlow_src/04_regression_all.R")
source("/data/salomonis2/LabFiles/Kyle/Development/infinityFlow/R/mod_infinityFlow_src/05_dimensionality_reduction.R")
source("/data/salomonis2/LabFiles/Kyle/Development/infinityFlow/R/mod_infinityFlow_src/06_converting_to_csv_and_FCS.R")
source("/data/salomonis2/LabFiles/Kyle/Development/infinityFlow/R/mod_infinityFlow_src/07_meaningplots.R")
source("/data/salomonis2/LabFiles/Kyle/Development/infinityFlow/R/mod_infinityFlow_src/08_background_correction.R")
source("/data/salomonis2/LabFiles/Kyle/Development/infinityFlow/R/mod_infinityFlow_src/misc.R")

path_flow_data <- "/data/salomonis2/LabFiles/Kyle/Analysis/2022_08_24_infinity_flow_benchmarking/input/original_infinity_flow_paper/mouse_lung_steady_state/backbone_subset_100"
backbone_selection_file <- file.path("/data/salomonis2/LabFiles/Kyle/Development/infinityFlow/R/backbone_selection_file.csv")

target_names <- iso_anno$Name
names(target_names) <- iso_anno$File

# Isotype to InfinityMarker annotation
isotypes <- iso_anno$Isotype
names(isotypes) <- iso_anno$File



path_to_output <- file.path(path_wd, "output/")
input_events_downsampling <- Inf
prediction_events_downsampling <- 10000
cores = 12L
path_to_intermediary_results <- file.path(path_wd, "intermediary_results/")



imputed_data <- infinity_flow(
  path_to_fcs = path_flow_data,
  path_to_output = path_to_output,
  path_to_intermediary_results = path_to_intermediary_results,
  backbone_selection_file = backbone_selection_file,
  annotation = target_names,
  isotype = isotypes,
  input_events_downsampling = input_events_downsampling,
  prediction_events_downsampling = prediction_events_downsampling,
  verbose = T,
  cores = cores,
  time_file_prefix = prefix_input
)