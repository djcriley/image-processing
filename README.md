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
  
