#!/bin/bash
##SBATCH --partition=compute   ### Partition
#SBATCH --job-name=Naive ### Job Name
#SBATCH --time=03:00:00     ### WallTime
#SBATCH --nodes=1          ### Number of Nodes
#SBATCH --tasks-per-node=1 ### Number of tasks (MPI processes=nodes*tasks-per-node. In this case, 32.

for sigma in 0.6 1.1
do
	for size in 1024 2048 4096 7680 10240 12800
	do
			for ((n=0;n<15;n++))
			do
				srun ./GPU_CannyEdge ~/lennas/Lenna_org_$size.pgm $sigma >> naive.csv
			done
	done
done
