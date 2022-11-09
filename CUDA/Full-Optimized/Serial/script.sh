#!/bin/bash
##SBATCH --partition=compute   ### Partition
#SBATCH --job-name=Serial ### Job Name
#SBATCH --time=03:00:00     ### WallTime
#SBATCH --nodes=1          ### Number of Nodes
#SBATCH --tasks-per-node=1 ### Number of tasks (MPI processes=nodes*tasks-per-node. In this case, 32.
      
for size in 256 512 1024 2048 4096 8192 10240
do       
		for ((n=0;n<30;n++))
		do
			srun ./canny_stage1 ~/lennas/Lenna_org_$size.pgm 0.6 >> serial.csv
		done
done
