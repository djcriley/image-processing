#!/bin/bash
#SBATCH --partition=compute   ### Partition
#SBATCH --job-name=Project02 ### Job Name
#SBATCH --time=03:00:00     ### WallTime
#SBATCH --nodes=2           ### Number of Nodes
#SBATCH --tasks-per-node=4 ### Number of tasks (MPI processes=nodes*tasks-per-node. In this case, 32.

for sigma in 0.6 1.1
do      
	for value in 1024 2048 4096 7680 10240 12800
	do 
		# for statistical runs
		for ((n=0;n<10;n++)) 
		do 
			srun ./canny_edge2 /exports/home/criley/images/Lenna_org_$value.pgm $sigma 4 >> parallel8.csv
		done
	done
done
