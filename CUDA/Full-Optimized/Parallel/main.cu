#include <cuda.h>
#include<stdlib.h>
#include "image_template.h"
#include <sys/time.h>
#include <string.h>
#include <time.h>
#include<math.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

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

	extern __shared__ float Ashared[];
	
	int localidx=threadIdx.x;
	int localidy=threadIdx.y;
	int globalidx=localidx+blockIdx.x*blockDim.x;
	int globalidy=localidy+blockIdx.y*blockDim.y;
	
	Ashared[localidx*blockDim.x+localidy] = image[globalidx*width+globalidy];
	__syncthreads();
	
	float sum;
	int offsetrow;
	int offsetcol;
	sum = 0;
	for(int kerrow = 0; kerrow < ker_h; kerrow++){
	for(int kercol = 0; kercol < ker_w; kercol++){
		offsetrow = -1 * (ker_h/2) + kerrow;
		offsetcol = -1 * (ker_w/2) + kercol;
		// in shared memory
		if((localidx + offsetrow) >= 0 && (localidx + offsetrow) < blockDim.x && (localidy + offsetcol) >= 0 && (localidy + offsetcol) <  blockDim.y) {
			sum = sum + Ashared[(offsetrow + localidx) * blockDim.x + (offsetcol + localidy)] * mask[(kerrow * ker_w) + kercol];
		}
		// not in shared memory
		else if ((globalidx + offsetrow) >= 0 && (globalidx + offsetrow) < width && (globalidy + offsetcol) >= 0 && (globalidy + offsetcol) <  height){
			sum = sum + image[(offsetrow + globalidx) * width + (offsetcol + globalidy)] * mask[(kerrow * ker_w) + kercol];
		}
	}
	}
	output[(globalidx * width) + globalidy] = sum;

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
__global__
void GPUsppression(float *magnitude, float *theta, int height, int width, float *supp) {

	extern __shared__ float Ashared[];
	
	int i=threadIdx.x;
	int j=threadIdx.y;
	int gi=i+blockIdx.x*blockDim.x;
	int gj=j+blockIdx.y*blockDim.y;
	
	Ashared[i*blockDim.x+j] = magnitude[gi*width+gj];
	__syncthreads();
	
	if(theta[gi * width + gj] < 0){
		theta[gi * width +gj] = theta[gi * width +gj] + M_PI;
		theta[gi * width +gj] = (180/M_PI) * theta[gi * width +gj];
	}
	
	// shared memory
	if(gi >= 0 && gi < width && gj >= 0 && gj <  height){
		// top bottom
		if(theta[gi * width + gj] <= 22.5 || theta[gi * width + gj] > 157.5){
			// top is i - 1, j
			if(i-1 >= 0){
				if(Ashared[i * blockDim.y + j] < Ashared[(i-1) * blockDim.y + j]){
					supp[gi * width + gj] = 0;
				}
			}
			else{
				if(gi-1 >= 0){
					if(magnitude[gi * width + gj] < magnitude[(gi-1) * width + gj]){
							supp[gi * width + gj] = 0;
					}
				}
			}
				
			// bottom is i + 1, j
			if(i+1 < blockDim.x){
				if(Ashared[i * blockDim.y + j] < Ashared[(i+1) * blockDim.y + j]){
					supp[gi * width + gj] = 0;
				}
			}
			else {
				if(gi+1 < height){
					if(magnitude[gi * width + gj] < magnitude[(gi+1) * width + gj]){
							supp[gi * width + gj] = 0;
					}
				}
			}
		}
		// top right bottom left
		else if(theta[gi * width + gj] > 22.5 && theta[gi * width + gj] < 67.5){
			// top right is i + 1, j + 1
			if(i + 1 < blockDim.x && j + 1 < blockDim.y){
				if(Ashared[i * blockDim.y + j] < Ashared[(i+1) * blockDim.y + j+1]){
					supp[gi * width + gj] = 0;
				}
			}
			else{
				if(gi + 1 < height && gj + 1 < width){
					if(magnitude[gi * width + gj] < magnitude[(gi+1) * width + gj+1]){
							supp[gi * width + gj] = 0;
					}
				}
			}
			// bottom left is i - 1, j + 1
			if(i - 1 >= 0 && j + 1 < blockDim.x){
				if(Ashared[i * blockDim.y + j] < Ashared[(i-1) * blockDim.y + j+1]){
					supp[gi * width + gj] = 0;
				}
			}
			else{
				if(gi - 1 >= 0 && gj + 1 < width){
					if(magnitude[gi * width + gj] < magnitude[(gi-1) * width + gj+1]){
							supp[gi * width + gj] = 0;
					}
				}
			}
		}
		// left right
		else if(theta[gi * width + gj] > 67.5 && theta[gi * width + gj] <= 112.5){
			// left is i, j-1
			if(j-1  >= 0){
				if(Ashared[i * blockDim.y + j] < Ashared[i * blockDim.y + j-1]){
					supp[gi * width + gj] = 0;
				}
			}
			else{
				if(gj-1  >= 0){
					if(magnitude[gi * width + gj] < magnitude[gi * width + gj-1]){
							supp[gi * width + gj] = 0;
					}
				}
			}
			// right is i, j+1
			if(j+1 < blockDim.y){
				if(Ashared[i * blockDim.y + j] < Ashared[i * blockDim.y + j+1]){
					supp[gi * width + gj] = 0;
				}
			}
			else{
				if(gj+1 < height){
					if(magnitude[gi * width + gj] < magnitude[gi * width + gj+1]){
							supp[gi * width + gj] = 0;
					}
				}
			}
		}

		// top left bottom right
		else if(theta[gi * width + gj] > 112.5 && theta[gi * width + gj] <= 157.5){
			// top left is i-1, j-1
			if(i-1 >= 0 && j-1 >= 0){
				if(Ashared[i * blockDim.y + j] < Ashared[(i-1) * blockDim.y + j-1]){
					supp[gi * width + gj] = 0;
				}
				
			}
			else{
				if(gi-1 >= 0 && gj-1 >= 0){
					if(magnitude[gi * width + gj] < magnitude[(gi-1) * width + gj-1]){
							supp[gi * width + gj] = 0;
					}
				}
			}
			// bottom right is i+1, j+1
			if(i+1 < blockDim.x && j+1 < blockDim.y){
				if(Ashared[i * blockDim.y + j] < Ashared[(i+1) * blockDim.y + j+1]){
					supp[gi * width + gj] = 0;
				}
			}
			else{
				if(gi+1 < height && gj+1 < height){
					if(magnitude[gi * width + gj] < magnitude[(gi+1) * width + gj+1]){
							supp[gi * width + gj] = 0;
					}
				}
			}
		}
	}
}
__global__
void GPUhysteresis(float *suppres, int height, int width, float *output, int t_hi, int t_low){
	
	// replace for loops with block ids
	int i = threadIdx.x + blockIdx.x*blockDim.x; //i index
	int j = threadIdx.y + blockIdx.y*blockDim.y; //j index
	
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
__global__
void GPUlinking(float *thres, int height, int width, float *output) {

	int i = threadIdx.x + blockIdx.x*blockDim.x; //i index
	int j = threadIdx.y + blockIdx.y*blockDim.y; //j index
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

int main(int argc, char **argv)
{
	
	int GPU_NO = 989312175%4;
	cudaSetDevice(GPU_NO); //GPU_NO = Your_Pacific_ID%4
	
    float *image;
    int height, width;
    
	//GPU timers
	struct timeval start, stop;
	
	int blocksize = atoi(argv[3]);
    float sigma = atof(argv[2]);
    char *filepath = argv[1];

    // read image into variables
    read_image_template(filepath, &image, &height, &width);
        
    float *edges;
	float *d_ker, *d_gder, *d_image, *d_tempHor, *d_Hor, *d_tempVert, *d_Vert, *d_mag, *d_dir, *d_supp, *d_hyst, *d_link, *d_sorted;
	
    
    
    // host malloc output files
    edges = (float *)malloc(sizeof(float)*width*height);
    
    
    
	cudaMalloc((void **)&d_image,sizeof(float)*width*height); // image
	cudaMalloc((void **)&d_tempHor,sizeof(float)*width*height); // temp horizontal
	cudaMalloc((void **)&d_Hor,sizeof(float)*width*height); // horizontal
	cudaMalloc((void **)&d_tempVert,sizeof(float)*width*height); // temp vertical
	cudaMalloc((void **)&d_Vert,sizeof(float)*width*height); // vertical
    cudaMalloc((void **)&d_mag,sizeof(float)*width*height); // magnitude
	cudaMalloc((void **)&d_dir,sizeof(float)*width*height); // direction
	cudaMalloc((void **)&d_supp,sizeof(float)*width*height); // supression
	cudaMalloc((void **)&d_hyst,sizeof(float)*width*height); // hysteresis
	cudaMalloc((void **)&d_link,sizeof(float)*width*height); // linking
	cudaMalloc((void **)&d_sorted,sizeof(float)*width*height); // sorted
    
    
    // start comp timer
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
	dim3 dimblock(blocksize, blocksize, 1);
	dim3 dimGrid(height/dimblock.x,width/dimblock.y);
	
	
	// impelement GPU kernel
	// call GPU kernel to run on block of threads
	// GPUconvolve(float *image, int height, int width, float *mask, int ker_h,int ker_w, float *output)
	
	// temp horizontal
	GPUconvolve<<<dimGrid,dimblock,sizeof(float)*dimblock.x*dimblock.y>>>(d_image,width,height, d_ker, w, 1, d_tempHor);
	cudaDeviceSynchronize();
	
	// horizontal
	GPUconvolve<<<dimGrid,dimblock,sizeof(float)*dimblock.x*dimblock.y>>>(d_tempHor,width,height, d_gder, 1, w, d_Hor);
	cudaDeviceSynchronize();
	
	// temp vertical
	GPUconvolve<<<dimGrid,dimblock,sizeof(float)*dimblock.x*dimblock.y>>>(d_image,width,height, d_ker, 1, w, d_tempVert);
	cudaDeviceSynchronize();
	
	// vertical
	GPUconvolve<<<dimGrid,dimblock,sizeof(float)*dimblock.x*dimblock.y>>>(d_tempVert,width,height, d_gder, w, 1, d_Vert);
	cudaDeviceSynchronize();
	
	// GPUMagDir(float *horizontal, float *vertical, int height, int width, float *mag, float *dir)
	GPUMagDir<<<dimGrid,dimblock>>>(d_Hor, d_Vert, height, width, d_mag, d_dir);
	cudaDeviceSynchronize();

	// suppression
	// Suppression(float *magnitude, float *direction, int height, int width, float *output)
	
	cudaMemcpy(d_supp,d_mag,sizeof(float)*width*height,cudaMemcpyDeviceToDevice);
	
	GPUsppression<<<dimGrid,dimblock,sizeof(float)*dimblock.x*dimblock.y>>>(d_mag, d_dir, height, width, d_supp);
	cudaDeviceSynchronize();
	
	cudaMemcpy(d_sorted,d_supp,sizeof(float)*width*height,cudaMemcpyDeviceToDevice);
	
	// GPU thrust sorting
	thrust::device_ptr<float> thr_d(d_sorted);

	thrust::device_vector<float>d_hyst_vec(thr_d,thr_d+(height*width));

	thrust::sort(d_hyst_vec.begin(),d_hyst_vec.end());

	int index = (int)(0.9*height*width);

	int t_hi = d_hyst_vec[index];

	int t_low = t_hi*0.2;
	
	
	// Hysteresis
	// GPUhysteresis(float *suppres, int height, int width, float *output, int threads, int t_hi, int t_low)
	GPUhysteresis<<<dimGrid,dimblock>>>(d_supp, height, width, d_hyst, t_hi, t_low);
	cudaDeviceSynchronize();
	
	// Edge linking
    // GPUlinking(float *thres, int height, int width, float *output)
    
	cudaMemcpy(d_link,d_hyst,sizeof(float)*width*height,cudaMemcpyDeviceToDevice);
    
	GPUlinking<<<dimGrid,dimblock>>>(d_hyst, height, width, d_link);
	cudaDeviceSynchronize();
    
    // memcopy results only images
	cudaMemcpy(edges,d_link,sizeof(float)*width*height,cudaMemcpyDeviceToHost);
	
	
	cudaDeviceSynchronize(); //wait for memcpy to finish
    
    // end comp timer
	gettimeofday(&stop, NULL);
	double comptime = ((stop.tv_sec * 1000000 + stop.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));
	
	//host writes the results
	char output_file[] = "edges.pgm";
    write_image_template(output_file, edges, width, height);
    
    // changed to double
	printf("%d, %d, %f\n",height, blocksize, comptime);
	
	
    // free pointers on host
	free(image);
    free(gausskernel);
    free(gaussderiv);
	free(edges);
	
	// free device arrays
	cudaFree(d_image);
	cudaFree(d_ker);
	cudaFree(d_gder);
	cudaFree(d_Hor);
	cudaFree(d_tempHor);
	cudaFree(d_Vert);
	cudaFree(d_tempVert);
	cudaFree(d_mag);
	cudaFree(d_dir);
	cudaFree(d_supp);
	cudaFree(d_hyst);
	cudaFree(d_link);
	cudaFree(d_sorted);
	
}
