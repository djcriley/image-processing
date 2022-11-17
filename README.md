# image-processing
This is an canny edge detector image processing project using C and parallel processing libraries like MPI, Cuda, Open MP. 

## Skills Used
- Makefiles
- Bash Scripting
- OpenMP
- Cuda
- MPI
- Analytics using Microsoft Excel

## Documentation
Each folder contains a different method to parallelize the serial implementation of the canny edge detector algorithm that was developed. The code was run using the bash scripts in order to automate the process of collecting data on different image sizes, and amount of resources allocated to the parallelization techniques. Makefiles are used to ensure that the code is compliled correctly. This project was tested using the lenna testing images with resolutions of: 1024, 2048, 4096, 7680, 10240, 12800. 

## Project structure
- Implementation Method (MPI, Cuda, Open MP)
  - Parallel
    - MakeFile
    - C source code File
    - Header file
    - runtimes CSV
    - script to run code
  - Serial
    - MakeFile
    - C source code File
    - Header file
    - runtimes CSV
    - script to run code
    
## Calculated Speedup Times
Speed up times calculated by timed parallel computation times divided by serial computation times. This excludes file read and writes in order to focus on speed up from parallelized functions.

![Screenshot 2022-11-17 at 3 14 07 PM](https://user-images.githubusercontent.com/43011353/202580009-b3bb3193-7ce5-4680-ab81-f078b90353f0.png)

![Screenshot 2022-11-17 at 3 09 45 PM](https://user-images.githubusercontent.com/43011353/202579905-9810a64c-7589-4b1d-b70d-024807b0bfbb.png)

![Screenshot 2022-11-16 at 9 47 25 PM](https://user-images.githubusercontent.com/43011353/202579913-f79c78d1-e8e4-499f-a5de-bd2ef4b36e0f.png)


  
