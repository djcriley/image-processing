#include <cuda.h>
#include<stdlib.h>
#include "image_template.h"
#include <sys/time.h>
#include <string.h>
#include <time.h>
#include<math.h>

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

__global__
void GPUconvolve(float *image, int height, int width, float *mask, int ker_h,int ker_w, float *output) {
	float sum;
	int offsetrow;
	int offsetcol;
	// replace outer for loops with block ids
	int i = threadIdx.x + blockIdx.x*blockDim.x; //i index
	int j = threadIdx.y + blockIdx.y*blockDim.y; //j index
	sum = 0;
	for(int kerrow = 0; kerrow < ker_h; kerrow++){
	for(int kercol = 0; kercol < ker_w; kercol++){
		offsetrow = -1 * (ker_h/2) + kerrow;
		offsetcol = -1 * (ker_w/2) + kercol;
		if(offsetrow + i < height && offsetcol + j < width && offsetrow + i >= 0 && offsetcol + j >= 0){
			sum = sum + image[(offsetrow + i) * width + offsetcol + j] * mask[(kerrow * ker_w) + kercol];
		}
	}
	}
	output[(i * width) + j] = sum;

}
__global__
void GPUMagDir(float *horizontal, float *vertical, int height, int width, float *mag, float *dir) {
	// replace for loops with block ids
	int i = threadIdx.x + blockIdx.x*blockDim.x; //i index
	int j = threadIdx.y + blockIdx.y*blockDim.y; //j index
	//magnitude
	mag[i * width + j] = sqrt( (vertical[i * width + j] * vertical[i * width + j]) + (horizontal[i * width + j] * horizontal[i * width + j]));
	// direction
	dir[i * width + j] = atan2(horizontal[i * width + j], vertical[i * width + j]);

}

int main(int argc, char **argv)
{
	// end to end timer
	struct timeval startend, stopend;
	gettimeofday(&startend, NULL);
	
	int GPU_NO = 989312175%4;
	cudaSetDevice(GPU_NO); //GPU_NO = Your_Pacific_ID%4
	
    float *image;
    int height, width;
    
	//GPU timers
	struct timeval gpustart, gpustop, convstart, convend, start, stop;
	
    float sigma = atof(argv[2]);
    char *filepath = argv[1];

    // read image into variables
    read_image_template(filepath, &image, &height, &width);
        
    float *magnitude, *direction;
	float *d_ker, *d_gder, *d_image, *d_tempHor, *d_Hor, *d_tempVert, *d_Vert, *d_mag, *d_dir;
    
    
    // host malloc output files
    magnitude = (float *)malloc(sizeof(float)*width*height);
    direction = (float *)malloc(sizeof(float)*width*height);
    
	cudaMalloc((void **)&d_image,sizeof(float)*width*height); // image
	cudaMalloc((void **)&d_tempHor,sizeof(float)*width*height); // temp horizontal
	cudaMalloc((void **)&d_Hor,sizeof(float)*width*height); // horizontal
	cudaMalloc((void **)&d_tempVert,sizeof(float)*width*height); // temp vertical
	cudaMalloc((void **)&d_Vert,sizeof(float)*width*height); // vertical
    cudaMalloc((void **)&d_mag,sizeof(float)*width*height); // magnitude
	cudaMalloc((void **)&d_dir,sizeof(float)*width*height); // direction
   
	gettimeofday(&start, NULL);
    // get gaussian kernels
	float *gausskernel;
    float *gaussderiv;
    int w;
    Gaussian(sigma, &gausskernel, &w);
    GaussianDeriv(sigma, &gaussderiv, &w);
    
	// cuda malloc image convultion magnitude and direction arrays
	cudaMalloc((void **)&d_ker,sizeof(float)*w); // gaussian kernel
	cudaMalloc((void **)&d_gder,sizeof(float)*w); // gaussian derivative
	
    
    
    //host transfers data to GPU for processing
    cudaMemcpy(d_image,image,sizeof(float)*width*height,cudaMemcpyHostToDevice);
	cudaMemcpy(d_ker,gausskernel,sizeof(float)*w,cudaMemcpyHostToDevice);
	cudaMemcpy(d_gder,gaussderiv,sizeof(float)*w,cudaMemcpyHostToDevice);
	
    
    
	//	create blocks and grid
	dim3 dimblock(32, 32, 1);
	dim3 dimGrid(height/dimblock.x,width/dimblock.y);
	
	
	
	// gpu calculations start
	gettimeofday(&gpustart,NULL);

	// impelement GPU kernel
	// call GPU kernel to run on block of threads
	// GPUconvolve(float *image, int height, int width, float *mask, int ker_h,int ker_w, float *output)
	
	// conv fucntions
	gettimeofday(&convstart, NULL);
	
	// temp horizontal
	GPUconvolve<<<dimGrid,dimblock>>>(d_image,width,height, d_ker, w, 1, d_tempHor);
	// horizontal
	GPUconvolve<<<dimGrid,dimblock>>>(d_tempHor,width,height, d_gder, 1, w, d_Hor);
	
	
	// temp vertical
	GPUconvolve<<<dimGrid,dimblock>>>(d_image,width,height, d_ker, 1, w, d_tempVert);
	// vertical
	GPUconvolve<<<dimGrid,dimblock>>>(d_tempVert,width,height, d_gder, w, 1, d_Vert);
	
	gettimeofday(&convend, NULL);
	
	
	// GPUMagDir(float *horizontal, float *vertical, int height, int width, float *mag, float *dir)
	GPUMagDir<<<dimGrid,dimblock>>>(d_Hor, d_Vert, height, width, d_mag, d_dir);
	
	

	
	double convtime = ((convend.tv_sec * 1000000 + convend.tv_usec) - (convstart.tv_sec * 1000000 + convstart.tv_usec));

	cudaDeviceSynchronize();
    
   
	//transfer results
    
    // memcopy results only images
	cudaMemcpy(magnitude,d_mag,sizeof(float)*width*height,cudaMemcpyDeviceToHost);
	cudaMemcpy(direction,d_dir,sizeof(float)*width*height,cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize(); //wait for memcpy to finish
	
	gettimeofday(&gpustop,NULL);
	
    double gputime = (gpustop.tv_usec/1000+gpustop.tv_sec*1000) - (gpustart.tv_usec/1000+gpustart.tv_sec*1000);
	
	gettimeofday(&stop, NULL);
	double comptime = ((stop.tv_sec * 1000000 + stop.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));
	
	//host writes the results
	char mag_file[] = "magnitude.pgm";
    write_image_template(mag_file, magnitude, width, height);
    char dir_file[] = "direction.pgm";
    write_image_template(dir_file, direction, width, height);
    
    gettimeofday(&stopend, NULL);
	double endtoendtime = ((stopend.tv_sec * 1000000 + stopend.tv_usec) - (startend.tv_sec * 1000000 + startend.tv_usec));
	
	// reverted back to int
	printf("%d, %3.2f, %f, %f, %f, %f\n",height, sigma, comptime, endtoendtime, convtime, gputime);
	
	
    // free pointers on host
	free(image);
    free(gausskernel);
    free(gaussderiv);
	free(magnitude);
    free(direction);
}
