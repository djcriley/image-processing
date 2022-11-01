#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "image_template.h"

#include<stdio.h>
#include<math.h>
#include<stdlib.h>

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
void convolve(float *image, int height, int width, float *mask, int ker_h,int ker_w, float *output) {
	float sum;
	int offsetrow;
	int offsetcol;
	for(int row = 0; row < height; row++){
	for(int col = 0; col < width; col++){
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
void MagDir(float *horizontal, float *vertical, int height, int width, float *mag, float *dir) {
	for(int i = 0; i < height; i++){
	for(int j = 0; j < width; j++){
		//magnitude
		mag[i * width + j] = sqrt( (vertical[i * width + j] * vertical[i * width + j]) + (horizontal[i * width + j] * horizontal[i * width + j]));
		// direction
		dir[i * width + j] = atan2(horizontal[i * width + j], vertical[i * width + j]);
    	}
     	}
}

void Suppression(float *magnitude, float *theta, int height, int width, float *supp) {
	
	memcpy(supp, magnitude, sizeof(float)*height*width);
	
	for(int i = 0; i < height; i++){
	for(int j = 0; j < width; j++){
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
void Hysteresis(float *suppres, int height, int width, float *output){
	// copy then sort suppression
	float *sorted;
	sorted = (float *)malloc((height * width)* sizeof(float));
	memcpy(sorted, suppres, sizeof(float)*height*width);
	qsort(sorted, height*width, sizeof(float), cmpfunc);
	
	// 90% is tHigh
	// 20 is tlow
	int tHigh = (int)(height * width * .90);
	//printf("%d\n", tHigh);
	// printf("%d\n", height*width);
	int t_hi = sorted[tHigh];
	int t_low = t_hi/5;
	// printf("%d\n", t_hi);
	// printf("%d\n", t_low);
	for(int i = 0; i < height; i++){
	for(int j = 0; j < width; j++){
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
	free(sorted);
}

void Linking(float *thres, int height, int width, float *output) {
	/*
	Edges = Hyst
	o For all pixels in Hyst image equal to 125
	o If a pixel is connected to any 255 intensity in 3x3 neighborhood, then set it the pixel to 255 in Edges image
	o else if no connection to any 255 intensity in 3x3 neighborhood, then set it to 0 in Edges image
	*/
	memcpy(output, thres, sizeof(float)*height*width);
	
	for(int i = 0; i < height; i++){
	for(int j = 0; j < width; j++){
	
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

    // for end to end time
    struct timeval stop, start;

    float *image;
    int height, width;

    float sigma = atof(argv[2]);
    char *filepath = argv[1];

    // read image into variables
    read_image_template(filepath, &image, &height, &width);
	float *temp_horizontal, *temp_vertical, *vertical, *horizontal, *magnitude, *direction;
	temp_horizontal = (float *)malloc((height * width)* sizeof(float));
	horizontal = (float *)malloc((height * width)* sizeof(float));
	temp_vertical = (float *)malloc((height * width)* sizeof(float));
	vertical = (float *)malloc((height * width)* sizeof(float));
	magnitude = (float *)malloc((height * width)* sizeof(float));
	direction = (float *)malloc((height * width)* sizeof(float));
	float *suppres;
	float *link;
	float *hyst;
	suppres = (float *)malloc((height * width)* sizeof(float));
	hyst = (float *)malloc((height * width)* sizeof(float));
	link = (float *)malloc((height * width)* sizeof(float));
	
    gettimeofday(&start, NULL);

    float *gausskernel;
    float *gaussderiv;
    int w;
    Gaussian(sigma, &gausskernel, &w);
    GaussianDeriv(sigma, &gaussderiv, &w);

    // horizontal gradient
    // temp
    // convolve(float *image, int height, int width, float *mask, int ker_h,int ker_w, float *output)
    // temp horizontal
    convolve(image, height, width, gausskernel, w, 1, temp_horizontal);
    // horizontal
    convolve(temp_horizontal, height, width, gaussderiv, 1, w, horizontal);
    //vertical temp
    convolve(image, height, width, gausskernel, 1, w, temp_vertical);
    // vertical
    convolve(temp_vertical, height, width, gaussderiv, w, 1, vertical);
    // magnitude and direction
    MagDir(horizontal, vertical, height, width, magnitude, direction);
    // suppression hysteria and linking
    Suppression(magnitude, direction, height, width, suppres);
	Hysteresis(suppres, height, width, hyst);
    Linking(hyst, height, width, link);
    

    gettimeofday(&stop, NULL);
    int comptime = ((stop.tv_sec * 1000000 + stop.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;

    
	char link_file[] = "linking.pgm";
	write_image_template(link_file, link, width, height);
	

    printf("%d, %d\n",height, comptime);

    // free all pointers
    free(image);
	free(filepath);
    free(gausskernel);
    free(gaussderiv);
    free(temp_horizontal);
    free(horizontal);
    free(temp_vertical);
    free(vertical);
    free(magnitude);
    free(direction);
	free(suppres);
	free(hyst);
	free(link);
	
    return 0;

}
