DIR=$(pwd)

cat <<EOF
#BSUB -L /bin/bash
#BSUB -W 24:00
#BSUB -n 12
#BSUB -R "span[ptile=4]"
#BSUB -M 250000
#BSUB -e $DIR/logs/%J.err
#BSUB -o $DIR/logs/%J.out

cd $DIR

# module load anaconda3
# conda activate /data/salomonis2/LabFiles/Kyle/Env/pyInfinityFlow_dev

pyInfinityFlow --data_dir /data/salomonis2/LabFiles/Kyle/Analysis/2022_08_24_infinity_flow_benchmarking/input/original_infinity_flow_paper/mouse_lung_steady_state/backbone/ \
--out_dir /data/salomonis2/LabFiles/Kyle/Development/pyInfinityFlow/test_scripts/mouse_lung_dataset/subset_50_features/output/ \
--backbone_annotation /data/salomonis2/LabFiles/Kyle/Development/pyInfinityFlow/data/mouse_lung_dataset_annotations/backbone_annotation_mouse_lung_dataset.csv \
--infinity_marker_annotation /data/salomonis2/LabFiles/Kyle/Development/pyInfinityFlow/data/mouse_lung_dataset_annotations/infinitymarker_annotation_50_feature_subset.csv \
--use_logicle_scaling True \
--normalization_method zscore \
--n_events_train 0 \
--n_events_validate 0 \
--ratio_for_validation 0.5 \
--n_events_combine 10000 \
--add_umap True \
--make_feature_plots True \
--save_h5ad True \
--save_feather True \
--save_file_handler True \
--save_regression_models True \
--verbosity 3 \
--n_cores 12

EOF

# ./<file_name> | bsub

