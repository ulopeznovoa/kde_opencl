/**************

opencl_util.c

Functions for creating/deallocating/managing OpenCL data structures

This source file is part of the set of programs used to derive climate
models' performance indices using multidimensional kernel-based
probability density functions, as described by:

Multi-objective climate model evaluation by means of multivariate kernel 
density estimators: efficient and multi-core implementations, by
Unai Lopez-Novoa, Jon Saenz, Alexander Mendiburu, Jose Miguel-Alonso, 
Inigo Errasti, Agustin Ezcurra, Gabriel Ibarra-Berastegi, 2014.


Copyright (c) 2014, Unai Lopez-Novoa, Jon Saenz, Alexander Mendiburu 
and Jose Miguel-Alonso  (from Universidad del Pais Vasco/Euskal 
		    Herriko Unibertsitatea)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Universidad del Pais Vasco/Euskal 
      Herriko Unibertsitatea  nor the names of its contributors may be 
      used to endorse or promote products derived from this software 
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

***************/

#include "opencl_util.h"

/*
size_t getMaxWorkGroupSize (cl_context ctx, cl_kernel ker)
{
    int err;
    // Find the maximum work group size
    size_t retSize = 0;
    size_t maxGroupSize = 0;
    // we must find the device asociated with this context first
    cl_device_id devid;   // we create contexts with a single device only
    err = clGetContextInfo (ctx, CL_CONTEXT_DEVICES, sizeof(devid), &devid, &retSize);
    //CL_CHECK_ERROR(err);
    if (retSize < sizeof(devid))  // we did not get any device, pass 0 to the function
       devid = 0;
    err = clGetKernelWorkGroupInfo (ker, devid, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
                &maxGroupSize, &retSize);
    //CL_CHECK_ERROR(err);
    return (maxGroupSize);
} */

inline const char *CLErrorString(cl_int err)
{
    switch (err)
    {
      case CL_SUCCESS:                         return "CL_SUCCESS";                         // break;
      case CL_DEVICE_NOT_FOUND:                return "CL_DEVICE_NOT_FOUND";                // break;
      case CL_DEVICE_NOT_AVAILABLE:            return "CL_DEVICE_NOT_AVAILABLE";            // break;
      case CL_COMPILER_NOT_AVAILABLE:          return "CL_COMPILER_NOT_AVAILABLE";          // break;
      case CL_MEM_OBJECT_ALLOCATION_FAILURE:   return "CL_MEM_OBJECT_ALLOCATION_FAILURE";   // break;
      case CL_OUT_OF_RESOURCES:                return "CL_OUT_OF_RESOURCES";                // break;
      case CL_OUT_OF_HOST_MEMORY:              return "CL_OUT_OF_HOST_MEMORY";              // break;
      case CL_PROFILING_INFO_NOT_AVAILABLE:    return "CL_PROFILING_INFO_NOT_AVAILABLE";    // break;
      case CL_MEM_COPY_OVERLAP:                return "CL_MEM_COPY_OVERLAP";                // break;
      case CL_IMAGE_FORMAT_MISMATCH:           return "CL_IMAGE_FORMAT_MISMATCH";           // break;
      case CL_IMAGE_FORMAT_NOT_SUPPORTED:      return "CL_IMAGE_FORMAT_NOT_SUPPORTED";      // break;
      case CL_BUILD_PROGRAM_FAILURE:           return "CL_BUILD_PROGRAM_FAILURE";           // break;
      case CL_MAP_FAILURE:                     return "CL_MAP_FAILURE";                     // break;
      case CL_INVALID_VALUE:                   return "CL_INVALID_VALUE";                   // break;
      case CL_INVALID_DEVICE_TYPE:             return "CL_INVALID_DEVICE_TYPE";             // break;
      case CL_INVALID_PLATFORM:                return "CL_INVALID_PLATFORM";                // break;
      case CL_INVALID_DEVICE:                  return "CL_INVALID_DEVICE";                  // break;
      case CL_INVALID_CONTEXT:                 return "CL_INVALID_CONTEXT";                 // break;
      case CL_INVALID_QUEUE_PROPERTIES:        return "CL_INVALID_QUEUE_PROPERTIES";        // break;
      case CL_INVALID_COMMAND_QUEUE:           return "CL_INVALID_COMMAND_QUEUE";           // break;
      case CL_INVALID_HOST_PTR:                return "CL_INVALID_HOST_PTR";                // break;
      case CL_INVALID_MEM_OBJECT:              return "CL_INVALID_MEM_OBJECT";              // break;
      case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"; // break;
      case CL_INVALID_IMAGE_SIZE:              return "CL_INVALID_IMAGE_SIZE";              // break;
      case CL_INVALID_SAMPLER:                 return "CL_INVALID_SAMPLER";                 // break;
      case CL_INVALID_BINARY:                  return "CL_INVALID_BINARY";                  // break;
      case CL_INVALID_BUILD_OPTIONS:           return "CL_INVALID_BUILD_OPTIONS";           // break;
      case CL_INVALID_PROGRAM:                 return "CL_INVALID_PROGRAM";                 // break;
      case CL_INVALID_PROGRAM_EXECUTABLE:      return "CL_INVALID_PROGRAM_EXECUTABLE";      // break;
      case CL_INVALID_KERNEL_NAME:             return "CL_INVALID_KERNEL_NAME";             // break;
      case CL_INVALID_KERNEL_DEFINITION:       return "CL_INVALID_KERNEL_DEFINITION";       // break;
      case CL_INVALID_KERNEL:                  return "CL_INVALID_KERNEL";                  // break;
      case CL_INVALID_ARG_INDEX:               return "CL_INVALID_ARG_INDEX";               // break;
      case CL_INVALID_ARG_VALUE:               return "CL_INVALID_ARG_VALUE";               // break;
      case CL_INVALID_ARG_SIZE:                return "CL_INVALID_ARG_SIZE";                // break;
      case CL_INVALID_KERNEL_ARGS:             return "CL_INVALID_KERNEL_ARGS";             // break;
      case CL_INVALID_WORK_DIMENSION:          return "CL_INVALID_WORK_DIMENSION";          // break;
      case CL_INVALID_WORK_GROUP_SIZE:         return "CL_INVALID_WORK_GROUP_SIZE";         // break;
      case CL_INVALID_WORK_ITEM_SIZE:          return "CL_INVALID_WORK_ITEM_SIZE";          // break;
      case CL_INVALID_GLOBAL_OFFSET:           return "CL_INVALID_GLOBAL_OFFSET";           // break;
      case CL_INVALID_EVENT_WAIT_LIST:         return "CL_INVALID_EVENT_WAIT_LIST";         // break;
      case CL_INVALID_EVENT:                   return "CL_INVALID_EVENT";                   // break;
      case CL_INVALID_OPERATION:               return "CL_INVALID_OPERATION";               // break;
      case CL_INVALID_GL_OBJECT:               return "CL_INVALID_GL_OBJECT";               // break;
      case CL_INVALID_BUFFER_SIZE:             return "CL_INVALID_BUFFER_SIZE";             // break;
      case CL_INVALID_MIP_LEVEL:               return "CL_INVALID_MIP_LEVEL";               // break;
      case CL_INVALID_GLOBAL_WORK_SIZE:        return "CL_INVALID_GLOBAL_WORK_SIZE";        // break;
      case CL_INVALID_PROPERTY:                return "CL_INVALID_PROPERTY";                // break;
      default:                                 return "UNKNOWN";                            // break;
  }
}

int openClSetup(oclVars * opencl, char * programName, int prefPlat, char prefDev, int dim)
{
	cl_int status; //Err handling
	
	//////// DEVICE SELECTION
	
	//Variables for retrieving the name
	char device_name[128];
	char vendor_name[128];	

	cl_platform_id platform_id;
	cl_device_id device_id; 

	//Get the OpenCL platform in the system, e.g. Intel
	if((prefPlat == -1) || (prefPlat == 0))// Use default
	{
		status = clGetPlatformIDs(1, &platform_id, NULL);
		if (CL_SUCCESS != status) {printf("Error: No OpenCL Platforms detected!\n"); return 1;	};
	}
	else
	{
		int plat_number = prefPlat + 1;
		cl_platform_id * platform_vector = (cl_platform_id *)malloc(sizeof(cl_platform_id) * plat_number);

		unsigned int plat_number_ret;
		status = clGetPlatformIDs(plat_number, platform_vector, &plat_number_ret);
		if (CL_SUCCESS != status) { printf("Error: Something gone wrong fetching OpenCL Platform %d \n",prefPlat); return 1;	};

		if(plat_number_ret < plat_number)
		{
			printf("\nWARNING: OpenCL Platform %d not found,turning to default\n",prefPlat);
			
			status = clGetPlatformIDs(1, &platform_id, NULL);
			if (CL_SUCCESS != status) {printf("Error: No OpenCL Platforms detected!\n"); return 1;	};
		}
		else
		{
			platform_id = platform_vector[prefPlat];
		}

		free(platform_vector);
	}

	//Get the platform name
	status = clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, 128 * sizeof(char),vendor_name,NULL);
	if (CL_SUCCESS != status) 	{ printf("Error retrieving platform name \n"); return 1;} 

	//Retrieve the platform default device, e.g. Multicore CPU
	cl_device_type dev_type;
	unsigned int num_devices;	
	
	switch(prefDev)
	{
		case 'C':
			dev_type = CL_DEVICE_TYPE_CPU;
			break; 
		case 'G':
			dev_type = CL_DEVICE_TYPE_GPU;
			break; 			
		case 'A':
			dev_type = CL_DEVICE_TYPE_ACCELERATOR;
			break; 			
		case 'D':
			dev_type = CL_DEVICE_TYPE_DEFAULT;
			break; 						
  
		default :
			dev_type = CL_DEVICE_TYPE_DEFAULT;
	}
		
	status = clGetDeviceIDs(platform_id, dev_type, 1, &device_id, &num_devices);
	//if(status != CL_SUCCESS) { printf("\nError: clGetDeviceIDs : %d\n",status); return 1; }	
			
	if(num_devices == 0)
	{
		printf("\nWARNING: %c type device not found in %s platform, using default",prefDev,vendor_name);

		status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
		if(status != CL_SUCCESS) { printf("\nError: clGetDeviceIDs : %d\n",status); return 1; }			
	}		
			
	//Get the device name		
	status = clGetDeviceInfo(device_id, CL_DEVICE_NAME, 100, &device_name, NULL);
	if (CL_SUCCESS != status) 	{	printf("\nError: clGetDeviceInfo : %d\n",status); return 1; 	};											
				
	//Show running device			
	printf("Running in %s %s \n",vendor_name,device_name);	

	//Set workgroup size for device: Half of the max allowed WG size
	status = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &opencl->wg_size, NULL);
	if (CL_SUCCESS != status) 	{	printf("\nError: clGetDeviceInfo WG Size : %d\n",status); return 1; 	};	

	opencl->wg_size /= 2;	

	opencl->scan_local_wsize  = 256;
	opencl->scan_global_wsize = 16384; // i.e. 64 work groups

	//////// OPENCL CONTEXT SET UP
	
	//Create an OpenCL context
	opencl->context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &status);
	if(status != CL_SUCCESS) { printf("\nError: clCreateContext : %d\n",status); return 1; }
		 
	//Create a command queue
	opencl->command_queue = clCreateCommandQueue(opencl->context, device_id, 0, &status);		
	if(status != CL_SUCCESS) { printf("\nError: command_queue : %d\n",status); return 1; }

	//////// READ AND COMPILE KERNELS

	char *source_str_kern;
	size_t source_size_kern;
	
	FILE * fp = fopen(programName, "r");
	if (!fp) { fprintf(stderr, "Failed to load kernel.\n"); exit(1); }
	source_str_kern = (char*)malloc(MAX_SOURCE_SIZE);
	source_size_kern = fread( source_str_kern, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );	
	
	//Create a program from the kernel source
	opencl->program = clCreateProgramWithSource(opencl->context, 1, (const char **)&source_str_kern, (const size_t *)&source_size_kern, &status);
	if(status != CL_SUCCESS) { 	printf("\nError: clCreateProgramWithSource (): %d\n",status); return 1;	}
 
	//Convert dim to string, as param for the kernel compiler
	//char params[30] = "-I ../";
	char params[64] = " -D DIM=";
	//char params[64] = "-Werror -D DIM=";
	//char params[64] = "-cl-opt-disable -D DIM=";
	char dim_string[3]; //Number of dimension in a string. Used as flag for the compiler
	sprintf(dim_string, "%d", dim); 
	strcat(params,dim_string); 	
 
	//Build the program
	status = clBuildProgram(opencl->program, 1, &device_id, params, NULL, NULL);
	if(status != CL_SUCCESS) 
	{ 			
		//Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(opencl->program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		// Allocate memory for the log
		char *log = (char *) malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(opencl->program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		printf("%s\n", log);		
			
		free(log);
		return 1;
	}

	opencl->kernel_calc_gp_2D = clCreateKernel(opencl->program, "calc_gp_2D", &status);    
	if(status != CL_SUCCESS) {printf("\nError compiling density kernel"); return 1;}   
	opencl->kernel_density_2D = clCreateKernel(opencl->program, "density_2D", &status);    
	if(status != CL_SUCCESS) {printf("\nError compiling density 2D kernel"); return 1;}   
							
	opencl->kernel_calc_slices = clCreateKernel(opencl->program, "calc_slices", &status);    
	if(status != CL_SUCCESS) {printf("\nError compiling calc_slices kernel"); return 1;}   
	opencl->kernel_calc_gp = clCreateKernel(opencl->program, "calc_gp", &status);    
	if(status != CL_SUCCESS) {printf("\nError compiling calc_gp kernel"); return 1;}   
	opencl->kernel_density = clCreateKernel(opencl->program, "density", &status);    
	if(status != CL_SUCCESS) {printf("\nError compiling density kernel"); return 1;}   

	opencl->kernel_reduce = clCreateKernel(opencl->program, "reduce", &status);
	if(status != CL_SUCCESS) {printf("\nError compiling reduce kernel"); return 1;}   
	opencl->kernel_top_scan = clCreateKernel(opencl->program, "top_scan", &status);
	if(status != CL_SUCCESS) {printf("\nError compiling top_scan kernel"); return 1;}   
	opencl->kernel_bottom_scan = clCreateKernel(opencl->program, "bottom_scan", &status);
	if(status != CL_SUCCESS) {printf("\nError compiling bottom_scan kernel"); return 1;}   

	//Scan kernels require at least 256 thread-workgroup
/*	if ( getMaxWorkGroupSize(opencl->context, reduce) < 256 || getMaxWorkGroupSize(opencl->context, top_scan) < 256 ||
         	getMaxWorkGroupSize(opencl->context, bottom_scan) < 256) 
	{
		printf("Scan requires a device that supports a work group size of at least 256");	
		return 1;
	}
*/

	return 0;
}

int createInitialBuffers(oclVars * opencl, int dim, MPDFEstimatorPtr mpdf, PDFPtr pdf, MAT *Sm1, MAT *eigenvectors, double *bounds, double *x0, 
	double *x1, double *dx, size_t * pdfcumsize, croppingVars * cp, int nsamples_pad4, int chunk_size, int max_slices_sample, int max_gp_sample)
{		
	cl_int status;
	int i;

	//INPUT STRUCTURES

   	opencl->samples = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mpdf->current * dim * sizeof(double), mpdf->X, &status); 
	if(status != CL_SUCCESS) { printf("\nError: clCreateBuffer samples: %d\n",status);	return 1;	}   
	opencl->samples_pc = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mpdf->current * dim * sizeof(double), mpdf->P, &status); 
	if(status != CL_SUCCESS) { printf("\nError: clCreateBuffer samples_pc: %d\n",status);	return 1;	}   
	opencl->eigenvectors = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dim * dim * sizeof(double), &eigenvectors->me[0][0], &status); 
	if(status != CL_SUCCESS) { printf("\nError: clCreateBuffer eigenvectors: %d\n",status);	return 1;	}   
	opencl->bounds = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dim * sizeof(double), bounds, &status);
	if(status != CL_SUCCESS) { printf("\nError: clCreateBuffer bounds: %d\n",status);	return 1;	}   
	opencl->x0 = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dim * sizeof(double), x0, &status); 
	if(status != CL_SUCCESS) { printf("\nError: clCreateBuffer x0: %d\n",status);	return 1;	}    
	opencl->x1 = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dim * sizeof(double), x1, &status); 
	if(status != CL_SUCCESS) { printf("\nError: clCreateBuffer x1: %d\n",status);	return 1;	}       
	opencl->dx = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dim * sizeof(double), dx, &status); 
	if(status != CL_SUCCESS) { printf("\nError: clCreateBuffer dx: %d\n",status);	return 1;	}   	
	
	if(dim >= 3)
	{	
		opencl->sm1 = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dim * dim * sizeof(double), &Sm1->me[0][0], &status); 
		if(status != CL_SUCCESS) { printf("\nError: clCreateBuffer sm1: %d\n",status);	return 1;	}   
		opencl->cp = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(croppingVars), cp, &status); 
		if(status != CL_SUCCESS) { printf("\nError: clCreateBuffer cp: %d\n",status);	return 1;	}   
	}

	//Convert pdfcumsize structure from size_t to int. Size_t has 4 bytes in AMD, and 8 bytes in Intel CPUs
	long * pdfcum_int = (long *)malloc(sizeof(long) * dim);
	for(i = 0; i < dim; i++)
		pdfcum_int[i] = (long)pdfcumsize[i];
	
	opencl->pdfcumsize = clCreateBuffer(opencl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dim * sizeof(long), pdfcum_int, &status); 
	if(status != CL_SUCCESS) { printf("\nError: clCreateBuffer pdfcumsize: %d\n",status);	return 1;	}   

	free(pdfcum_int);

	//OUTPUT STRUCTURES
	
	if(dim >= 3)
	{
		opencl->slices_sample = clCreateBuffer(opencl->context, CL_MEM_READ_WRITE, nsamples_pad4 * sizeof(int), NULL, &status); 
		if(status != CL_SUCCESS) { printf("\nError: clCreateBuffer slices_sample: %d\n",status);	return 1;	}   	
		opencl->slices_sample_scan = clCreateBuffer(opencl->context, CL_MEM_READ_WRITE, (nsamples_pad4 + 1) * sizeof(int), NULL, &status); 
		if(status != CL_SUCCESS) { printf("\nError: clCreateBuffer slices_sample_scan: %d\n",status);	return 1;	}   	
		opencl->slices_sample_lower = clCreateBuffer(opencl->context, CL_MEM_READ_WRITE, mpdf->current * sizeof(double), NULL, &status); 
		if(status != CL_SUCCESS) { printf("\nError: clCreateBuffer slices_sample_lower: %d\n",status);	return 1;	}   	

		if((nsamples_pad4 - mpdf->current) > 0)
		{
			int padding[3] = {0,0,0};
			int pad_elems = nsamples_pad4 - mpdf->current;
		
			status = clEnqueueWriteBuffer(opencl->command_queue, opencl->slices_sample, CL_TRUE, sizeof(int) * mpdf->current, sizeof(int) * pad_elems, padding, 0, NULL, NULL);
			if(status != CL_SUCCESS) { printf("\nError: clEnqueueWriteBuffer padding: %d\n",status);	return 1;	}   
		}
	}

	int size;

	if(dim >= 3)
		size = chunk_size * max_slices_sample;
	else
		size = mpdf->current; 

 	opencl->gp_x = clCreateBuffer(opencl->context, CL_MEM_READ_WRITE, size * sizeof(int), NULL, &status); 
	if(status != CL_SUCCESS) { printf("\nError: clCreateBuffer slices_sample: %d\n",status);	return 1;	}
 	opencl->gp_y = clCreateBuffer(opencl->context, CL_MEM_READ_WRITE, size * sizeof(int), NULL, &status); 
	if(status != CL_SUCCESS) { printf("\nError: clCreateBuffer slices_sample: %d\n",status);	return 1;	}
 	opencl->gp_xy = clCreateBuffer(opencl->context, CL_MEM_READ_WRITE, (size + 4) * sizeof(int), NULL, &status); 
	if(status != CL_SUCCESS) { printf("\nError: clCreateBuffer slices_sample: %d\n",status);	return 1;	}
 	opencl->gp_xy_scan = clCreateBuffer(opencl->context, CL_MEM_READ_WRITE, (size + 4) * sizeof(int), NULL, &status); 
	if(status != CL_SUCCESS) { printf("\nError: clCreateBuffer slices_sample: %d\n",status);	return 1;	}
 	opencl->gp_x_lower = clCreateBuffer(opencl->context, CL_MEM_READ_WRITE, size * sizeof(double), NULL, &status); 
	if(status != CL_SUCCESS) { printf("\nError: clCreateBuffer slices_sample: %d\n",status);	return 1;	}
 	opencl->gp_y_lower = clCreateBuffer(opencl->context, CL_MEM_READ_WRITE, size * sizeof(double), NULL, &status); 
	if(status != CL_SUCCESS) { printf("\nError: clCreateBuffer slices_sample: %d\n",status);	return 1;	}
	if(dim >= 3)
	{
		opencl->sample_index = clCreateBuffer(opencl->context, CL_MEM_READ_WRITE, size * sizeof(int), NULL, &status); 
		if(status != CL_SUCCESS) { printf("\nError: clCreateBuffer slices_sample: %d\n",status);	return 1;	}
		opencl->slice_index = clCreateBuffer(opencl->context, CL_MEM_READ_WRITE, size * sizeof(int), NULL, &status); 
		if(status != CL_SUCCESS) { printf("\nError: clCreateBuffer slices_sample: %d\n",status);	return 1;	}
	}

	size = chunk_size * max_gp_sample;

 	opencl->density_positions = clCreateBuffer(opencl->context, CL_MEM_READ_WRITE, size * sizeof(int), NULL, &status); 
	if(status != CL_SUCCESS) { printf("\nError: clCreateBuffer slices_sample: %d\n",status);	return 1;	}
 	opencl->density_values = clCreateBuffer(opencl->context, CL_MEM_READ_WRITE, size * sizeof(double), NULL, &status); 
	if(status != CL_SUCCESS) { printf("\nError: clCreateBuffer slices_sample: %d\n",status);	return 1;	}

	//Scan temp buffer
	size_t num_work_groups = opencl->scan_global_wsize / opencl->scan_local_wsize;   
	opencl->d_isums = clCreateBuffer(opencl->context, CL_MEM_READ_WRITE, num_work_groups * sizeof(int), NULL, &status);
	if(status != CL_SUCCESS) {printf("\nError creating d_isums buffers"); return 1;}   

	return 0;
}


int setKernelArgs(oclVars * opencl, int nsamples, int nsamples_pad4, double h2, double cd, int dim)
{
	int status;

	if(dim == 2)
	{

	//Kernel calc_gp_2D
	status  = clSetKernelArg(opencl->kernel_calc_gp_2D, 0, sizeof(cl_int), (void*)&nsamples);
	status |= clSetKernelArg(opencl->kernel_calc_gp_2D, 1, sizeof(cl_mem), (void*)&opencl->samples);
	status |= clSetKernelArg(opencl->kernel_calc_gp_2D, 2, sizeof(cl_mem), (void*)&opencl->x0);	
	status |= clSetKernelArg(opencl->kernel_calc_gp_2D, 3, sizeof(cl_mem), (void*)&opencl->x1);	
	status |= clSetKernelArg(opencl->kernel_calc_gp_2D, 4, sizeof(cl_mem), (void*)&opencl->dx);			
    	status |= clSetKernelArg(opencl->kernel_calc_gp_2D, 5, sizeof(cl_mem), (void*)&opencl->bounds);

	status |= clSetKernelArg(opencl->kernel_calc_gp_2D, 6, sizeof(cl_mem), (void*)&opencl->gp_x);
    	status |= clSetKernelArg(opencl->kernel_calc_gp_2D, 7, sizeof(cl_mem), (void*)&opencl->gp_y);
    	status |= clSetKernelArg(opencl->kernel_calc_gp_2D, 8, sizeof(cl_mem), (void*)&opencl->gp_xy);
   	status |= clSetKernelArg(opencl->kernel_calc_gp_2D, 9, sizeof(cl_mem), (void*)&opencl->gp_x_lower);
   	status |= clSetKernelArg(opencl->kernel_calc_gp_2D, 10, sizeof(cl_mem), (void*)&opencl->gp_y_lower);
 
	if(status != CL_SUCCESS) { 	printf("\nError: clSetKernelArg (kernel_calc_gp_2D): %d\n",status); return 1;	}			
	
	//Kernel density
	status  = clSetKernelArg(opencl->kernel_density_2D, 2, sizeof(cl_double), (void*)&cd);
	status |= clSetKernelArg(opencl->kernel_density_2D, 3, sizeof(cl_double), (void*)&h2);
	status |= clSetKernelArg(opencl->kernel_density_2D, 4, sizeof(cl_mem), (void*)&opencl->eigenvectors);	
	status |= clSetKernelArg(opencl->kernel_density_2D, 5, sizeof(cl_mem), (void*)&opencl->x0);	
	status |= clSetKernelArg(opencl->kernel_density_2D, 6, sizeof(cl_mem), (void*)&opencl->dx);		
	status |= clSetKernelArg(opencl->kernel_density_2D, 7, sizeof(cl_mem), (void*)&opencl->pdfcumsize);		

	status |= clSetKernelArg(opencl->kernel_density_2D, 8, sizeof(cl_mem), (void*)&opencl->samples_pc);		

	status |= clSetKernelArg(opencl->kernel_density_2D, 9, sizeof(cl_mem), (void*)&opencl->gp_x);
    	status |= clSetKernelArg(opencl->kernel_density_2D, 10, sizeof(cl_mem), (void*)&opencl->gp_y);
   	status |= clSetKernelArg(opencl->kernel_density_2D, 11, sizeof(cl_mem), (void*)&opencl->gp_x_lower);
   	status |= clSetKernelArg(opencl->kernel_density_2D, 12, sizeof(cl_mem), (void*)&opencl->gp_y_lower);
    	status |= clSetKernelArg(opencl->kernel_density_2D, 13, sizeof(cl_mem), (void*)&opencl->gp_xy_scan);
 
    	status |= clSetKernelArg(opencl->kernel_density_2D, 14, sizeof(cl_mem), (void*)&opencl->density_values);
    	status |= clSetKernelArg(opencl->kernel_density_2D, 15, sizeof(cl_mem), (void*)&opencl->density_positions);
	if(status != CL_SUCCESS) { 	printf("\nError: clSetKernelArg (kernel_density_2D): %d\n",status); return 1;	}			

	// SCAN

	size_t num_work_groups = opencl->scan_global_wsize / opencl->scan_local_wsize;   

	// Set the kernel arguments for the reduction kernel
	status  = clSetKernelArg(opencl->kernel_reduce, 0, sizeof(cl_mem), (void*)&opencl->gp_xy);
	status |= clSetKernelArg(opencl->kernel_reduce, 1, sizeof(cl_mem), (void*)&opencl->d_isums);
	status |= clSetKernelArg(opencl->kernel_reduce, 2, sizeof(cl_int), (void*)&nsamples_pad4);
	status |= clSetKernelArg(opencl->kernel_reduce, 3, opencl->scan_local_wsize * sizeof(int), NULL);

	// Set the kernel arguments for the top-level scan
	status |= clSetKernelArg(opencl->kernel_top_scan, 0, sizeof(cl_mem), (void*)&opencl->d_isums);
	status |= clSetKernelArg(opencl->kernel_top_scan, 1, sizeof(cl_int), (void*)&num_work_groups);
	status |= clSetKernelArg(opencl->kernel_top_scan, 2, opencl->scan_local_wsize * 2 * sizeof(int), NULL);

	    // Set the kernel arguments for the bottom-level scan
	status |= clSetKernelArg(opencl->kernel_bottom_scan, 0, sizeof(cl_mem), (void*)&opencl->gp_xy);
	status |= clSetKernelArg(opencl->kernel_bottom_scan, 1, sizeof(cl_mem), (void*)&opencl->d_isums);
	status |= clSetKernelArg(opencl->kernel_bottom_scan, 2, sizeof(cl_mem), (void*)&opencl->gp_xy_scan);
	status |= clSetKernelArg(opencl->kernel_bottom_scan, 3, sizeof(cl_int), (void*)&nsamples_pad4);
	status |= clSetKernelArg(opencl->kernel_bottom_scan, 4, opencl->scan_local_wsize * 2 * sizeof(int), NULL);

	if(status != CL_SUCCESS) { 	printf("\nError: clSetKernelArg scan %d\n",status); return 1;	}		

	}
	else if(dim >= 3)
	{	

	//Kernel calc_slices 
	status  = clSetKernelArg(opencl->kernel_calc_slices, 0, sizeof(cl_int), (void*)&nsamples);
	status |= clSetKernelArg(opencl->kernel_calc_slices, 1, sizeof(cl_mem), (void*)&opencl->samples);
	status |= clSetKernelArg(opencl->kernel_calc_slices, 2, sizeof(cl_mem), (void*)&opencl->slices_sample);
	status |= clSetKernelArg(opencl->kernel_calc_slices, 3, sizeof(cl_mem), (void*)&opencl->slices_sample_lower);
	status |= clSetKernelArg(opencl->kernel_calc_slices, 4, sizeof(cl_mem), (void*)&opencl->x0);
	status |= clSetKernelArg(opencl->kernel_calc_slices, 5, sizeof(cl_mem), (void*)&opencl->x1);
	status |= clSetKernelArg(opencl->kernel_calc_slices, 6, sizeof(cl_mem), (void*)&opencl->dx);
	status |= clSetKernelArg(opencl->kernel_calc_slices, 7, sizeof(cl_mem), (void*)&opencl->bounds);
	if(status != CL_SUCCESS) {printf("\nError clSetKernelArg opencl->kernel_calc_slices"); return 1;}  
	 
	//Kernel calc_gp
	status  = clSetKernelArg(opencl->kernel_calc_gp, 1, sizeof(cl_mem), (void*)&opencl->cp);
	status |= clSetKernelArg(opencl->kernel_calc_gp, 2, sizeof(cl_mem), (void*)&opencl->sm1);
	status |= clSetKernelArg(opencl->kernel_calc_gp, 3, sizeof(cl_mem), (void*)&opencl->x0);	
	status |= clSetKernelArg(opencl->kernel_calc_gp, 4, sizeof(cl_mem), (void*)&opencl->x1);	
	status |= clSetKernelArg(opencl->kernel_calc_gp, 5, sizeof(cl_mem), (void*)&opencl->dx);		
	
    	status |= clSetKernelArg(opencl->kernel_calc_gp, 6, sizeof(cl_mem), (void*)&opencl->samples);
    	status |= clSetKernelArg(opencl->kernel_calc_gp, 7, sizeof(cl_mem), (void*)&opencl->slices_sample);
    	status |= clSetKernelArg(opencl->kernel_calc_gp, 8, sizeof(cl_mem), (void*)&opencl->slices_sample_scan);
   	status |= clSetKernelArg(opencl->kernel_calc_gp, 9, sizeof(cl_mem), (void*)&opencl->slices_sample_lower);
 
	status |= clSetKernelArg(opencl->kernel_calc_gp, 10, sizeof(cl_mem), (void*)&opencl->gp_x);
    	status |= clSetKernelArg(opencl->kernel_calc_gp, 11, sizeof(cl_mem), (void*)&opencl->gp_y);
    	status |= clSetKernelArg(opencl->kernel_calc_gp, 12, sizeof(cl_mem), (void*)&opencl->gp_xy);
   	status |= clSetKernelArg(opencl->kernel_calc_gp, 13, sizeof(cl_mem), (void*)&opencl->gp_x_lower);
   	status |= clSetKernelArg(opencl->kernel_calc_gp, 14, sizeof(cl_mem), (void*)&opencl->gp_y_lower);
 
    	status |= clSetKernelArg(opencl->kernel_calc_gp, 15, sizeof(cl_mem), (void*)&opencl->sample_index);
    	status |= clSetKernelArg(opencl->kernel_calc_gp, 16, sizeof(cl_mem), (void*)&opencl->slice_index);
	if(status != CL_SUCCESS) { 	printf("\nError: clSetKernelArg (kernel_calc_gp): %d\n",status); return 1;	}			
	
	//Kernel density
	status  = clSetKernelArg(opencl->kernel_density, 1, sizeof(cl_double), (void*)&cd);
	status |= clSetKernelArg(opencl->kernel_density, 2, sizeof(cl_double), (void*)&h2);
	status |= clSetKernelArg(opencl->kernel_density, 3, sizeof(cl_mem), (void*)&opencl->eigenvectors);	
	status |= clSetKernelArg(opencl->kernel_density, 4, sizeof(cl_mem), (void*)&opencl->x0);	
	status |= clSetKernelArg(opencl->kernel_density, 5, sizeof(cl_mem), (void*)&opencl->dx);		
	status |= clSetKernelArg(opencl->kernel_density, 6, sizeof(cl_mem), (void*)&opencl->pdfcumsize);		

	status |= clSetKernelArg(opencl->kernel_density, 7, sizeof(cl_mem), (void*)&opencl->samples_pc);		
	status |= clSetKernelArg(opencl->kernel_density, 8, sizeof(cl_mem), (void*)&opencl->slices_sample_lower);

	status |= clSetKernelArg(opencl->kernel_density, 9, sizeof(cl_mem), (void*)&opencl->gp_x);
    	status |= clSetKernelArg(opencl->kernel_density, 10, sizeof(cl_mem), (void*)&opencl->gp_y);
   	status |= clSetKernelArg(opencl->kernel_density, 11, sizeof(cl_mem), (void*)&opencl->gp_x_lower);
   	status |= clSetKernelArg(opencl->kernel_density, 12, sizeof(cl_mem), (void*)&opencl->gp_y_lower);
    	status |= clSetKernelArg(opencl->kernel_density, 13, sizeof(cl_mem), (void*)&opencl->gp_xy_scan);
 
    	status |= clSetKernelArg(opencl->kernel_density, 14, sizeof(cl_mem), (void*)&opencl->sample_index);
    	status |= clSetKernelArg(opencl->kernel_density, 15, sizeof(cl_mem), (void*)&opencl->slice_index);
    	status |= clSetKernelArg(opencl->kernel_density, 16, sizeof(cl_mem), (void*)&opencl->density_values);
    	status |= clSetKernelArg(opencl->kernel_density, 17, sizeof(cl_mem), (void*)&opencl->density_positions);
	if(status != CL_SUCCESS) { 	printf("\nError: clSetKernelArg (kernel_density): %d\n",status); return 1;	}			
	
	// SCAN

	size_t num_work_groups = opencl->scan_global_wsize / opencl->scan_local_wsize;   

	// Set the kernel arguments for the reduction kernel
	status  = clSetKernelArg(opencl->kernel_reduce, 0, sizeof(cl_mem), (void*)&opencl->slices_sample);
	status |= clSetKernelArg(opencl->kernel_reduce, 1, sizeof(cl_mem), (void*)&opencl->d_isums);
	status |= clSetKernelArg(opencl->kernel_reduce, 2, sizeof(cl_int), (void*)&nsamples_pad4);
	status |= clSetKernelArg(opencl->kernel_reduce, 3, opencl->scan_local_wsize * sizeof(int), NULL);

	// Set the kernel arguments for the top-level scan
	status |= clSetKernelArg(opencl->kernel_top_scan, 0, sizeof(cl_mem), (void*)&opencl->d_isums);
	status |= clSetKernelArg(opencl->kernel_top_scan, 1, sizeof(cl_int), (void*)&num_work_groups);
	status |= clSetKernelArg(opencl->kernel_top_scan, 2, opencl->scan_local_wsize * 2 * sizeof(int), NULL);

	    // Set the kernel arguments for the bottom-level scan
	status |= clSetKernelArg(opencl->kernel_bottom_scan, 0, sizeof(cl_mem), (void*)&opencl->slices_sample);
	status |= clSetKernelArg(opencl->kernel_bottom_scan, 1, sizeof(cl_mem), (void*)&opencl->d_isums);
	status |= clSetKernelArg(opencl->kernel_bottom_scan, 2, sizeof(cl_mem), (void*)&opencl->slices_sample_scan);
	status |= clSetKernelArg(opencl->kernel_bottom_scan, 3, sizeof(cl_int), (void*)&nsamples_pad4);
	status |= clSetKernelArg(opencl->kernel_bottom_scan, 4, opencl->scan_local_wsize * 2 * sizeof(int), NULL);

	if(status != CL_SUCCESS) { 	printf("\nError: clSetKernelArg scan %d\n",status); return 1;	}		
	}	

	return 0;
}

//Due to a bug in the Intel OpenCL SDK, clSetKernelArg can not be applied to cl_mem structures that 
//have not been initialied

int freeOpenClSetup(oclVars * opencl, int dim)
{	
	cl_int status;	

	status  = clReleaseMemObject(opencl->samples);
        status |= clReleaseMemObject(opencl->samples_pc);
        status |= clReleaseMemObject(opencl->sm1);
        status |= clReleaseMemObject(opencl->eigenvectors);
        status |= clReleaseMemObject(opencl->bounds);
        status |= clReleaseMemObject(opencl->x0); 
        status |= clReleaseMemObject(opencl->x1); 
        status |= clReleaseMemObject(opencl->dx); 
        status |= clReleaseMemObject(opencl->cp); 
        status |= clReleaseMemObject(opencl->pdfcumsize);
        status |= clReleaseMemObject(opencl->gp_x);
        status |= clReleaseMemObject(opencl->gp_y);
        status |= clReleaseMemObject(opencl->gp_xy);
        status |= clReleaseMemObject(opencl->gp_xy_scan);
        status |= clReleaseMemObject(opencl->gp_x_lower);
        status |= clReleaseMemObject(opencl->gp_y_lower);
        status |= clReleaseMemObject(opencl->density_values);
        status |= clReleaseMemObject(opencl->density_positions);
        if(dim >= 3)
	{
		status |= clReleaseMemObject(opencl->slices_sample);
		status |= clReleaseMemObject(opencl->slices_sample_lower);
		status |= clReleaseMemObject(opencl->slices_sample_scan);
	 
		status |= clReleaseMemObject(opencl->sample_index);
		status |= clReleaseMemObject(opencl->slice_index);
	}
        status |= clReleaseMemObject(opencl->d_isums); 
	if(status != CL_SUCCESS) { 	printf("\nError: clReleaseMemObject: %d\n",status); return 1;	}		

	status  = clReleaseKernel(opencl->kernel_calc_gp_2D);
	status |= clReleaseKernel(opencl->kernel_density_2D);
	status |= clReleaseKernel(opencl->kernel_calc_slices);
	status |= clReleaseKernel(opencl->kernel_calc_gp);
	status |= clReleaseKernel(opencl->kernel_density);
	status |= clReleaseKernel(opencl->kernel_reduce);
	status |= clReleaseKernel(opencl->kernel_top_scan);
	status |= clReleaseKernel(opencl->kernel_bottom_scan);
	if(status != CL_SUCCESS) { 	printf("\nError: clReleaseKernel: %d\n",status); return 1;	}	
    
    	status = clReleaseCommandQueue(opencl->command_queue);
	if(status != CL_SUCCESS) { 	printf("\nError: clReleaseCommandQueue: %d\n",status); return 1;	}	
	
	status = clReleaseContext(opencl->context);	
	if(status != CL_SUCCESS) { 	printf("\nError: clReleaseContext: %d\n",status); return 1;	}			
	
	return 0;
}

