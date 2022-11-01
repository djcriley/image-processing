// Cooper Riley
// 1.0 version
// 1/15/21

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "image_template.h"
#include <mpi.h>

#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<omp.h>

#define M_PI 3.14159265358979323846

int cmpfunc (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}

void Gaussian(float sigma, float **mask, int *w){
    	int a = round(2.5 * sigma - 0.5);
	*w = 2 * a + 1;
	// dynamically allocate Mask array to the size of the kernel 
	(*mask) = (float *)malloc((*w) * sizeof(float));
	float sum = 0;
	for(int i = 0; i <= *w - 1; i++){
		(*mask)[i] = exp((-1*(i-a)*(i-a)) / (2*sigma*sigma));
		sum = sum + (*mask)[i];
	}
	for(int i = 0; i <= *w - 1; i++){
		(*mask)[i] = (*mask)[i] / sum;
	}
}
void GaussianDeriv(float sigma, float **mask, int *w) {

	int a = round(2.5 * sigma - 0.5);
	*w = 2 * a + 1;
	// dynamically allocate Mask array to the size of the kernel 
	(*mask) = (float *)malloc((*w) * sizeof(float));
	float sum = 0;
	for(int i = 0; i <= *w - 1; i++){
		(*mask)[i] = -1 * (i-a) * exp((-1*(i-a)*(i-a)) / (2*sigma*sigma));
		sum = sum - i * (*mask)[i];
	}
	
	for(int i = 0; i <= *w - 1; i++){
		(*mask)[i] = (*mask)[i] / sum;
	}
	// set up the flipping of the kernel dereivative
	for(int i = 0; i < (*w/2) -1; i++){
		float temp = (*mask)[*w-1-i];
		(*mask)[*w-1-i] = (*mask)[i];
		(*mask)[i] = temp;
	}
}
void convolve(float *image, int height, int width, float *mask, int ker_h,int ker_w, float *output, int threads) {
	float sum;
	int offsetrow;
	int offsetcol;
	int row;
	int col;
	// omp_set_num_threads(threads);
	// #pragma omp parallel for private(col, offsetrow, offsetcol, sum)
	for(row = 0; row < height; row++){
	for(col = 0; col < width; col++){
		sum = 0;
		for(int kerrow = 0; kerrow < ker_h; kerrow++){
		for(int kercol = 0; kercol < ker_w; kercol++){
			offsetrow = -1 * floor(ker_h/2) + kerrow;
			offsetcol = -1 * floor(ker_w/2) + kercol;
			if(offsetrow + row < height && offsetcol + col < width && offsetrow + row >= 0 && offsetcol + col >= 0){
				sum = sum + image[(offsetrow + row) * width + offsetcol + col] * mask[(kerrow * ker_w) + kercol];
			}
		}
		}
		output[(row * width) + col] = sum;
	}
	}
}
void MagDir(float *horizontal, float *vertical, int height, int width, float *mag, float *dir, int threads) {
	int i;
	int j;
	// omp_set_num_threads(threads);
	// #pragma omp parallel for private(j)
	for(i = 0; i < height; i++){
	for(j = 0; j < width; j++){
		//magnitude
		mag[i * width + j] = sqrt( (vertical[i * width + j] * vertical[i * width + j]) + (horizontal[i * width + j] * horizontal[i * width + j]));
		// direction
		dir[i * width + j] = atan2(horizontal[i * width + j], vertical[i * width + j]);
    	}
     	}
}

void Suppression(float *magnitude, float *theta, int height, int width, float *supp, int threads) {
	
	memcpy(supp, magnitude, sizeof(float)*height*width);
	int i;
	int j;
	// omp_set_num_threads(threads);
	// #pragma omp parallel for private(j)
	for(i = 0; i < height; i++){
	for(j = 0; j < width; j++){
		// first check in psuedo code
		if(theta[i * width + j] < 0){
			theta[i * width +j] = theta[i * width +j] + M_PI;
			theta[i * width +j] = (180/M_PI) * theta[i * width +j];
		}
		// top bottom
		if(theta[i * width + j] <= 22.5 || theta[i * width + j] > 157.5){
			// top is i - 1, j
			if(i-1 >= 0){
				if(magnitude[i * width + j] < magnitude[(i-1) * width + j]){
						supp[i * width + j] = 0;
				}
			}
			// bottom is i + 1, j 
			if(i+1 < height){
				if(magnitude[i * width + j] < magnitude[(i+1) * width + j]){
						supp[i * width + j] = 0;
				}
			}
			
		}
		// top right bottom left
		else if(theta[i * width + j] > 22.5 && theta[i * width + j] < 67.5){
			// top right is i + 1, j + 1
			if(i + 1 <= height && j + 1 < width){
				if(magnitude[i * width + j] < magnitude[(i+1) * width + j+1]){
						supp[i * width + j] = 0;
				}
			}
			// bottom left is i - 1, j + 1
			if(i - 1 >= 0 && j + 1 < width){
				if(magnitude[i * width + j] < magnitude[(i-1) * width + j+1]){
						supp[i * width + j] = 0;
				}
			}
		}
		// left right
		else if(theta[i * width + j] > 67.5 && theta[i * width + j] <= 112.5){
			// left is i, j-1
			if(j-1  >= 0){
				if(magnitude[i * width + j] < magnitude[i * width + j-1]){
						supp[i * width + j] = 0;
				}
			}
			// right is i, j+1
			if(j+1 <= height){
				if(magnitude[i * width + j] < magnitude[i * width + j+1]){
						supp[i * width + j] = 0;
				}
			}
		}
		// top left bottom right
		else if(theta[i * width + j] > 112.5 && theta[i * width + j] <= 157.5){
			// top left is i-1, j-1
			if(i-1 >= 0 && j-1 >= 0){
				if(magnitude[i * width + j] < magnitude[(i-1) * width + j-1]){
						supp[i * width + j] = 0;
				}
			}
			// bottom right is i+1, j+1
			if(i+1 <= height && j+1 <= height){
				if(magnitude[i * width + j] < magnitude[(i+1) * width + j+1]){
						supp[i * width + j] = 0;
				}
			}
		}
    	}
     	}
	
}
void Threshold(float *suppres, int height, int width, float *output, int threads, int t_hi, int t_low){
	/*
	// copy then sort suppression
	float *sorted;
	sorted = (float *)malloc((height * width)* sizeof(float));
	memcpy(sorted, suppres, sizeof(float)*height*width);
	// try to parrallize
	qsort(sorted, height*width, sizeof(float), cmpfunc);
	// 90% is tHigh
	// 20 is tlow
	int t_hi = sorted[(int)(height * width * .90)];
	int t_low = t_hi/5;
	*/
	int i;
	int j;
	// omp_set_num_threads(threads);
	// #pragma omp parallel for private(j)
	for(i = 0; i < height; i++){
	for(j = 0; j < width; j++){
		if(suppres[i * width + j] >= t_hi){
			output[i * width + j] = 255;
		}
		if(suppres[i * width + j] < t_hi && suppres[i * width + j] > t_low){
			output[i * width + j] = 125;
		}
		if(suppres[i * width + j] <= t_low){
			output[i * width + j] = 0;
		}
	}
	}
	// free(sorted);
}

void Hysteresis(float *thres, int height, int width, float *output, int threads) {
	/*
	Edges = Hyst
	o For all pixels in Hyst image equal to 125
	o If a pixel is connected to any 255 intensity in 3x3 neighborhood, then set it the pixel to 255 in Edges image
	o else if no connection to any 255 intensity in 3x3 neighborhood, then set it to 0 in Edges image
	*/
	memcpy(output, thres, sizeof(float)*height*width);
	int i;
	int j;
	// omp_set_num_threads(threads);
	// #pragma omp parallel for private(j)
	for(i = 0; i < height; i++){
	for(j = 0; j < width; j++){
		// for all 125 pixels
		if(thres[i * width + j] == 125){
			// if connected to 255 set pixel to 255 check all 6 possibilties around it in 3x3
			output[i * width + j] = 0;
			// top is i - 1, j
			if(i-1 >= 0){
				if(thres[(i-1) * width + j] == 255){
					output[i * width + j] = 255;
				}
			}
			// bottom is i + 1, j 
			if(i+1 <= height){
				if(thres[(i+1) * width + j] == 255){
					output[i * width + j] = 255;
				}
			}
			// top right is i + 1, j + 1
			if(i + 1 <= height && j + 1 < width){
				if(thres[(i+1) * width + j+1] == 255){
					output[i * width + j] = 255;
				}
			}
			// bottom left is i - 1, j + 1
			if(i - 1 >= 0 && j + 1 < width){
				if(thres[(i-1) * width + j+1] == 255){
					output[i * width + j] = 255;
				}
			}
			
			// left is i, j-1
			if(j-1  >= 0){
				if(thres[i * width + j-1] == 255){
					output[i * width + j] = 255;
				}
			}
			// right is i, j+1
			if(j+1 <= width){
				if(thres[i * width + j+1] == 255){
					output[i * width + j] = 255;
				}
			}
			// top left is i-1, j-1
			if(i-1 >= 0 && j-1 >= 0){
				if(thres[(i-1) * width + j-1] == 255){
					output[i * width + j] = 255;
				}
			}
			// bottom right is i+1, j+1
			if(i+1 < height && j+1 < width){
				if(thres[(i+1) * width + j+1] == 255){
					output[i * width + j] = 255;
				}
			}	
		}
	}
	}

}


int main( int argc, char *argv[]){

	//Initialize MPI
	MPI_Init(&argc,&argv);
	

	// for end to end time
	struct timeval stop, start;
	struct timeval startend, stopend;
	struct timeval convstart, convend;
	gettimeofday(&startend, NULL);

	float *image, *chunk, *hor_no_ghost, *horizontal, *chunk_th, *chunk_h;
	float *vert_no_ghost, *vertical, *chunk_tv, *chunk_v;
	float *chunk_mag, *chunk_dir, *magnitude, *direction, *mag, *dir;
	float *chunk_supp, *supp, *sorted, *chunk_thresh, *chunk_edge, *supp_no_extra;
	float *thresh, *edges, *thresh_with_extra;
	int height, width, comm_size, comm_rank;
	
	//int threads = atoi(argv[3]);
	int threads = 0;
	float sigma = atof(argv[2]);
	char *filepath = argv[1];
	
	// gaussian doesnt need to be included in MPI to start out
	float *gausskernel;
	float *gaussderiv;
	int w;
	Gaussian(sigma, &gausskernel, &w);
	GaussianDeriv(sigma, &gaussderiv, &w);
	
	int a = w/2;
	
	//Get the number of processors
	MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
	MPI_Comm_rank(MPI_COMM_WORLD,&comm_rank);
	
	

	if(comm_rank==0){
		read_image_template(filepath,&image,&width,&height);
	}
	
	MPI_Bcast(&width,1,MPI_INT,0,MPI_COMM_WORLD); // bcasting width to other procs 
	MPI_Bcast(&height,1,MPI_INT,0,MPI_COMM_WORLD); // bcasting height to other procs
	gettimeofday(&start, NULL);
	
	if (comm_rank==0)
	{
		chunk = (float *)malloc(sizeof(float)*width*((height/comm_size)+a));
	}
	else if (comm_rank==comm_size-1)
	{
		chunk = (float *)malloc(sizeof(float)*width*((height/comm_size)+a));
	}
	else{
		chunk = (float *)malloc(sizeof(float)*width*((height/comm_size)+(2*a)));
	}
	
	if (comm_rank==0)
	{
		horizontal = (float *)malloc(height*width* sizeof(float));
		vertical = (float *)malloc(height*width* sizeof(float));
		magnitude = (float *)malloc(height*width* sizeof(float));
		direction = (float *)malloc(height*width* sizeof(float));
		supp = (float *)malloc(height*width* sizeof(float));
		thresh = (float *)malloc(height*width* sizeof(float));
		edges = (float *)malloc(height*width* sizeof(float));
	}

	int *displs, *sendcnt;
	if(!comm_rank) 
	{
		sendcnt = (int *)malloc(sizeof(int)*comm_size);
		displs = (int *)malloc(sizeof(int)*comm_size);

		displs[0] = 0;
		sendcnt[0] = ((height/comm_size)+a)*width;

		for(int i=1;i<comm_size-1;i++) {
			displs[i] = ((height/comm_size)*i-a)*width;
			sendcnt[i] = ((height/comm_size)+2*a)*width;
		}

		displs[comm_size-1] = ((height/comm_size)*(comm_size-1)-a)*width;
		sendcnt[comm_size-1] = ((height/comm_size)+a)*width;
	}
	
	if(comm_rank==0)
	{
		MPI_Scatterv(image,sendcnt,displs,MPI_FLOAT,chunk,(height/comm_size+a)*width,MPI_FLOAT,0,MPI_COMM_WORLD);
	} 
	else if (comm_rank==comm_size-1)
	{
		MPI_Scatterv(image,sendcnt,displs,MPI_FLOAT,chunk,(height/comm_size+a)*width,MPI_FLOAT,0,MPI_COMM_WORLD); 
	}
	else 
	{
		MPI_Scatterv(image,sendcnt,displs,MPI_FLOAT,chunk,(height/comm_size+(2*a))*width,MPI_FLOAT,0,MPI_COMM_WORLD);
	} 
	
	// to time convolution
	gettimeofday(&convstart, NULL);
	// convolve(float *image, int height, int width, float *mask, int ker_h,int ker_w, float *output, int threads)
	
	
	if (comm_rank==0 || comm_rank == comm_size-1)
	{
		// temp horizontal
		chunk_th = (float *)malloc(sizeof(float)*width*((height/comm_size)+a));
		convolve(chunk, ((height/comm_size)+a), width, gausskernel, w, 1, chunk_th, 0);
		// temp vertical
		chunk_tv = (float *)malloc(sizeof(float)*width*((height/comm_size)+a));
		convolve(chunk, ((height/comm_size)+a), width, gausskernel, 1, w, chunk_tv, 0);
	}
	
	else
	{	
		// temp_horizontal
		chunk_th = (float *)malloc(sizeof(float)*width*((height/comm_size)+(2*a)));
		convolve(chunk, ((height/comm_size)+(2*a)), width, gausskernel, w, 1, chunk_th, 0);
		// temp vertical
		chunk_tv = (float *)malloc(sizeof(float)*width*((height/comm_size)+(2*a)));
		convolve(chunk, ((height/comm_size)+(2*a)), width, gausskernel, 1, w, chunk_tv, 0);
	}
	MPI_Gather(chunk_tv,width*(height/comm_size), MPI_FLOAT, vertical, width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	if(comm_rank==0)
	{
		MPI_Scatterv(vertical,sendcnt,displs,MPI_FLOAT,chunk,(height/comm_size+a)*width,MPI_FLOAT,0,MPI_COMM_WORLD);
	
	} 
	else if (comm_rank==comm_size-1)
	{
		MPI_Scatterv(vertical,sendcnt,displs,MPI_FLOAT,chunk,(height/comm_size+a)*width,MPI_FLOAT,0,MPI_COMM_WORLD);
	}	
	else 
	{
		MPI_Scatterv(vertical,sendcnt,displs,MPI_FLOAT,chunk,(height/comm_size+(2*a))*width,MPI_FLOAT,0,MPI_COMM_WORLD);
		
	}
	
	if (comm_rank==0 || comm_rank == comm_size-1)
	{
		// horizontal
		chunk_h = (float *)malloc(sizeof(float)*width*((height/comm_size)+a));
		convolve(chunk_th, ((height/comm_size)+a), width, gaussderiv, 1, w, chunk_h, 0);
		// vertical
		chunk_v = (float *)malloc(sizeof(float)*width*((height/comm_size)+a));
		convolve(chunk, ((height/comm_size)+a), width, gaussderiv, w, 1, chunk_v, 0);
		
	}
	else
	{	
		// horizontal
		chunk_h = (float *)malloc(sizeof(float)*width*((height/comm_size)+(2*a)));
		convolve(chunk_th, ((height/comm_size)+(2*a)), width, gaussderiv, 1, w, chunk_h, 0);
		// vertical
		chunk_v = (float *)malloc(sizeof(float)*width*((height/comm_size)+(2*a)));
		convolve(chunk, ((height/comm_size)+(2*a)), width, gaussderiv, w, 1, chunk_v, 0);
	
	}
	
	
	
	hor_no_ghost = (float *)malloc((height/comm_size)*width* sizeof(float));
	vert_no_ghost = (float *)malloc((height/comm_size)*width* sizeof(float));
	if (comm_rank==0)
	{
		memcpy(hor_no_ghost, chunk_h, sizeof(float)*(height/comm_size)*width);
		memcpy(vert_no_ghost, chunk_v, sizeof(float)*(height/comm_size)*width);
	}
	else
	{
		memcpy(hor_no_ghost, chunk_h+a*width, sizeof(float)*(height/comm_size)*width);
		memcpy(vert_no_ghost, chunk_v+a*width, sizeof(float)*(height/comm_size)*width);
	}
	
	gettimeofday(&convend, NULL);
	
	
	chunk_dir = (float *)malloc(sizeof(float)*width*(height/comm_size));
	chunk_mag = (float *)malloc(sizeof(float)*width*(height/comm_size));
	MagDir(hor_no_ghost, vert_no_ghost, (height/comm_size), width, chunk_mag, chunk_dir, 0);
	
	MPI_Gather(chunk_mag,width*(height/comm_size), MPI_FLOAT, magnitude, width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Gather(chunk_dir,width*(height/comm_size), MPI_FLOAT, direction, width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	
	// scatter mag and dir
	if (comm_rank==0)
	{
		mag = (float *)malloc(sizeof(float)*width*((height/comm_size)+a));
		dir = (float *)malloc(sizeof(float)*width*((height/comm_size)+a));
	}
	else if (comm_rank==comm_size-1)
	{
		mag = (float *)malloc(sizeof(float)*width*((height/comm_size)+a));
		dir = (float *)malloc(sizeof(float)*width*((height/comm_size)+a));
	}
	else{
		mag = (float *)malloc(sizeof(float)*width*((height/comm_size)+(2*a)));
		dir = (float *)malloc(sizeof(float)*width*((height/comm_size)+(2*a)));
	}
	if(comm_rank==0)
	{
		MPI_Scatterv(magnitude,sendcnt,displs,MPI_FLOAT,mag,(height/comm_size+a)*width,MPI_FLOAT,0,MPI_COMM_WORLD);
		MPI_Scatterv(direction,sendcnt,displs,MPI_FLOAT,dir,(height/comm_size+a)*width,MPI_FLOAT,0,MPI_COMM_WORLD);
	} 
	else if (comm_rank==comm_size-1)
	{
		MPI_Scatterv(magnitude,sendcnt,displs,MPI_FLOAT,mag,(height/comm_size+a)*width,MPI_FLOAT,0,MPI_COMM_WORLD);
		MPI_Scatterv(direction,sendcnt,displs,MPI_FLOAT,dir,(height/comm_size+a)*width,MPI_FLOAT,0,MPI_COMM_WORLD); 
	}
	else 
	{
		MPI_Scatterv(magnitude,sendcnt,displs,MPI_FLOAT,mag,(height/comm_size+(2*a))*width,MPI_FLOAT,0,MPI_COMM_WORLD);
		MPI_Scatterv(direction,sendcnt,displs,MPI_FLOAT,dir,(height/comm_size+(2*a))*width,MPI_FLOAT,0,MPI_COMM_WORLD);
	}
	
	// suppression 
	// Suppression(float *magnitude, float *theta, int height, int width, float *supp, int threads)
	if (comm_rank==0 || comm_rank == comm_size-1)
	{
		chunk_supp = (float *)malloc(sizeof(float)*width*((height/comm_size)+a));
		Suppression(mag, dir, ((height/comm_size)+a), width, chunk_supp, 0);
		
	}
	else
	{		
		chunk_supp = (float *)malloc(sizeof(float)*width*((height/comm_size)+(2*a)));
		Suppression(mag, dir, ((height/comm_size)+(2*a)), width, chunk_supp, 0);
	}
	
	supp_no_extra = (float *)malloc((height/comm_size)*width* sizeof(float));
	if (comm_rank==0)
	{
		memcpy(supp_no_extra, chunk_supp, sizeof(float)*(height/comm_size)*width);
	}
	else
	{
		memcpy(supp_no_extra, chunk_supp, sizeof(float)*(height/comm_size)*width);
	}
	
	MPI_Gather(supp_no_extra,width*(height/comm_size), MPI_FLOAT, supp, width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	float t_hi, t_low;
	if (comm_rank==0)
	{
		sorted = (float *)malloc((height * width)* sizeof(float));
		memcpy(sorted, supp, sizeof(float)*height*width);
		qsort(sorted, height*width, sizeof(float), cmpfunc);
		// 90% is tHigh
		// 20 is tlow
		t_hi = sorted[(int)(height * width * .90)];
		t_low = t_hi/5;
	}

	// bcast tlow and thi
	MPI_Bcast(&t_hi,1,MPI_INT,0,MPI_COMM_WORLD); // bcasting t_hi to other procs 
	MPI_Bcast(&t_low,1,MPI_INT,0,MPI_COMM_WORLD); // bcasting t_low to other procs
	
	// now do threshold and hysteresis
	chunk_thresh = (float *)malloc(sizeof(float)*width*(height/comm_size));
	Threshold(supp_no_extra, (height/comm_size), width, chunk_thresh, 0, t_hi, t_low);
	
	MPI_Gather(chunk_thresh,width*(height/comm_size), MPI_FLOAT, thresh, width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	
	
	if (comm_rank==0 || comm_rank == comm_size-1)
	{
		thresh_with_extra = (float *)malloc(sizeof(float)*width*((height/comm_size)+a));
		
	}
	else
	{	
		thresh_with_extra = (float *)malloc(sizeof(float)*width*((height/comm_size)+(2*a)));
		
	}
	if (comm_rank==0 || comm_rank == comm_size-1)
	{
		
		MPI_Scatterv(thresh,sendcnt,displs,MPI_FLOAT,thresh_with_extra,(height/comm_size+a)*width,MPI_FLOAT,0,MPI_COMM_WORLD); 
	}
	else
	{	
		
		MPI_Scatterv(thresh,sendcnt,displs,MPI_FLOAT,thresh_with_extra,((height/comm_size)+(2*a))*width,MPI_FLOAT,0,MPI_COMM_WORLD);
	}
	
	
	if (comm_rank==0 || comm_size-1)
	{
		chunk_edge = (float *)malloc(sizeof(float)*width*((height/comm_size)+a));
		Hysteresis(thresh_with_extra, ((height/comm_size)+a), width, chunk_edge, 0);
	}
	else
	{	
		chunk_edge = (float *)malloc(sizeof(float)*width*((height/comm_size)+(2*a)));
		Hysteresis(thresh_with_extra, ((height/comm_size)+(2*a)), width, chunk_edge, 0);
	}
	
	
	
	
	//MPI_Gather(hor_no_ghost,width*(height/comm_size), MPI_FLOAT, horizontal, width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
	//MPI_Gather(vert_no_ghost,width*(height/comm_size), MPI_FLOAT, vertical, width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
	//MPI_Gather(chunk_dir,width*(height/comm_size), MPI_FLOAT, direction, width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
	//MPI_Gather(chunk_mag,width*(height/comm_size), MPI_FLOAT, magnitude, width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
	//MPI_Gather(chunk_supp,width*(height/comm_size), MPI_FLOAT, supp, width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
	//MPI_Gather(chunk_thresh,width*(height/comm_size), MPI_FLOAT, thresh, width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Gather(chunk_edge,width*(height/comm_size), MPI_FLOAT, edges, width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
	gettimeofday(&stop, NULL);
	
	gettimeofday(&stopend, NULL);
	
		
	if(comm_rank==0){
		//write_image_template("horizontal.pgm", horizontal, width, height);
		//write_image_template("vertical.pgm", vertical, width, height);
		//write_image_template("magnitude.pgm", magnitude, width, height);
		//write_image_template("direction.pgm", direction, width, height);
		//write_image_template("supression.pgm", supp, width, height);
		//write_image_template("threshold.pgm", thresh, width, height);
		write_image_template("edges.pgm", edges, width, height);
		int endtoendtime = ((stopend.tv_sec * 1000000 + stopend.tv_usec) - (startend.tv_sec * 1000000 + startend.tv_usec)) / 1000;
		int comptime = ((stop.tv_sec * 1000000 + stop.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
		int convtime = ((convend.tv_sec * 1000000 + convend.tv_usec) - (convstart.tv_sec * 1000000 + convstart.tv_usec)) / 1000;
		printf("%d, %3.2f, %d, %d, %d, %d\n",height, sigma, threads, comptime, endtoendtime, convtime);
	}

	
	free(chunk);
	free(gausskernel);
	free(gaussderiv);
	free(chunk_th);
	free(chunk_h);
	free(hor_no_ghost);
	free(chunk_tv);
	free(chunk_v);
	free(vert_no_ghost);
	free(chunk_dir);
	free(chunk_mag);
	free(chunk_supp);
	free(mag);
	free(dir);
	free(chunk_edge);
	free(chunk_thresh);
	free(thresh_with_extra);
	if (comm_rank==0)
	{
		free(image);
		free(displs);
		free(sendcnt);
		free(horizontal);
		free(vertical);
		free(magnitude);
		free(direction);
		free(supp);
		free(sorted);
		free(thresh);
		free(edges);
	}
	
	MPI_Finalize();
	
	return 0;
}	

