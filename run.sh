tract=AF_left

# extract features from the vtp files to obtain input file of the model
# plase name the input folder including vtp files with their tract name
# please modify this file (and fibers.py) according to the name of your point-wise features
#python feat_read.py ./$tract ./feat_tract

# Train and test the model; the model with best performance will be saved in nets/$tract;
# the best r is saved in txt files in reports/$tract
# vtp files of tracts are read according to the subject id in teh .csv file; there will be errors if the vtp file of any subject in the .csv file is missing
python main.py -indir feat_tract -outdir $tract --CUDA_id 0 --task tpvt --dataset PointSet_pair --num_points 2048 --tracts $tract --channels 3 --net_architecture PointNet --mode train --epochs 200

#comparison experiments
#python main.py -indir feat_tract -outdir $tract --CUDA_id 0 --task tpvt --dataset PointSet --num_points 2048 --tracts $tract --channels 3 --net_architecture PointNet --mode train --epochs 200
