DIR=$(pwd)

cat <<EOF
#BSUB -L /bin/bash
#BSUB -W 72:00
#BSUB -n 12
#BSUB -R "span[ptile=4]"
#BSUB -M 250000
#BSUB -e $DIR/logs/%J.err
#BSUB -o $DIR/logs/%J.out

cd $DIR

# module load anaconda3
# conda activate /data/salomonis2/LabFiles/Kyle/Env/r_4_2_infinityflow

Rscript infinityFlow_r_mouse_lung_full_dataset.R

EOF

# ./<file_name> | bsub