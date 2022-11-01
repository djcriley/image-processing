#!/bin/bash
for ((n=0;n<30;n++))
do       
	for value in 1024 2048 4096 7680 10240 12800
	do       
		./canny_edge2 /exports/home/criley/images/Lenna_org_$value.pgm 0.6 >> serial.csv
	done
done
for ((n=0;n<30;n++))
do       
	for value in 1024 2048 4096 7680 10240 12800
	do       
		./canny_edge2 /exports/home/criley/images/Lenna_org_$value.pgm 1.1 >> serial.csv
	done
done
