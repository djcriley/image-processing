#!/bin/bash
#SBATCH --partition=compute   ### Partition
#SBATCH --job-name=CannyEdge ### Job Name
#SBATCH --time=08:00:00     ### WallTime
#SBATCH --nodes=1           ### Number of Nodes
#SBATCH --ntasks-per-node=1 ### Number of tasks (MPI processes)
for sigma in 0.6 1.1
do       
	for ((n=0;n<30;n++))
	do
		for value in 1024 2048 4096 7680 10240 12800
		do  
			for thread in 2 4 8 16
			do
				srun --nodes=1 ./canny_edge2 /exports/home/criley/images/Lenna_org_$value.pgm $sigma $thread >> parallel.csv
			done
		done
	done
done

