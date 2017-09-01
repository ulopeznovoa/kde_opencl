/**************

opencl_util.h

OpenCL data structure and function prototypes 

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

#ifndef __OPENCL__
#define __OPENCL__ 1

#define MAX_SOURCE_SIZE (0x100000)
#define ceil_int(a,b) (((a % b) == 0)? (a / b) : ((a / b) + 1)) //Ceiling for a and b integer

#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <opencl.h>
#else
#include <CL/cl.h>
#endif

#include "matrix.h"
#include "matrix2.h"
#include "MPDFEstimator.h"
#include "PDF.h"



//Struct that contains the variables for the cropping geometric calculations
//Used to copy their constant values to the OpenCL device
struct croppingVariables
{
	double cosTheta;
	double sinTheta;
	double X2;
	double Y2;
};
typedef struct croppingVariables croppingVars;

//Struct to hold
struct openClVars
{
	cl_context context;
	cl_command_queue command_queue;    
	
	cl_program program;
	
	size_t wg_size;

	size_t scan_local_wsize;
	size_t scan_global_wsize; // i.e. 64 work groups

	//// Kernels

	//2D
	cl_kernel kernel_calc_gp_2D;
	cl_kernel kernel_density_2D;

	//3D
	cl_kernel kernel_calc_slices;
	cl_kernel kernel_calc_gp;
	cl_kernel kernel_density;	

	//Scan
	cl_kernel kernel_reduce;
	cl_kernel kernel_top_scan;
	cl_kernel kernel_bottom_scan;

	//// Memory variables
	
	//Read Only
	cl_mem samples;
	cl_mem samples_pc;
	cl_mem sm1;
	cl_mem eigenvectors;
	cl_mem bounds;
	cl_mem x0;
	cl_mem x1;
	cl_mem dx;
	cl_mem cp;
	cl_mem pdfcumsize;
	
	//Read-Write
	cl_mem slices_sample;
	cl_mem slices_sample_lower;
	cl_mem slices_sample_scan;
	cl_mem gp_x;
	cl_mem gp_y;
	cl_mem gp_xy;
	cl_mem gp_xy_scan;
	cl_mem gp_x_lower;
	cl_mem gp_y_lower;
	cl_mem density_values;
	cl_mem density_positions;
	cl_mem sample_index;
	cl_mem slice_index;

	//Scan
	cl_mem d_isums;	
};

typedef struct openClVars oclVars;

//Due to a bug in the Intel OpenCL SDK, clSetKernelArg can not be applied to cl_mem structures that 
//have not been initialied

int openClSetup(oclVars * opencl, char * programName, int prefPlat, char prefDev, int dim);

int createInitialBuffers(oclVars * opencl, int dim, MPDFEstimatorPtr mpdf, PDFPtr pdf, MAT *Sm1, MAT *eigenvectors, 
  double *bounds, double *x0, double *x1, double *dx, size_t * pdfcumsize, croppingVars * cp, int nsamples_pad4, int chunk_size, int max_slices_sample, int max_gp_sample);
int setKernelArgs(oclVars * opencl, int nsamples, int nsamples_pad4, double h2, double cd, int dim);

int freeOpenClSetup(oclVars * opencl, int dim);

#endif
