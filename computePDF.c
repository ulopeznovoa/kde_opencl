/**************

computePDF.c

Functions used to actually compute the PDFs for several dimensionalities.

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

#include "computePDF.h"
#include "linalg.h"

double volumeConstant(int dim)
{
	if(dim == 1)
		return 2.;
	else if(dim == 2)
		return acos(-1.);
	else if (dim == 3)
		return acos(-1.)*4./3.;	
	else
		return unit_sphere_volume(dim);
}
 
/**** Functions to calculate the PDF of a defined 2D space (box) for a given sample. ****/ 
 
//Compute the density in the bounding box of a sample - Function for 2D spaces
void compute2DBox_2D(PDFPtr pdf, double * PC, double * lower, int * tot_ev_per_dim, double * gridpoint, size_t * dif_pos, 
	double * x0, double * dx, double h2, double cd, MAT * eigenvectors, double *  densValues, int *  densPosition)
{
	int u,v,l; //Loop variables
	double temp; //Will contain the absolute distance value from gridpoint to sample.
	double PCdot[2] __attribute__((aligned(64)));
	
	for(gridpoint[0] = lower[0], u = 0; u < tot_ev_per_dim[0]; gridpoint[0] += dx[0], u++)
	{  
		int HalfPosition = (((gridpoint[0] - x0[0])/ dx[0]) * pdf->pdfcumsize[0]);
				
		//Compiler flag to inform about structure alignment		
		//__assume_aligned(densValues,64);
		//__assume_aligned(densPosition,64);				
				
		#ifdef _OPENMP
		#pragma simd private(PCdot,temp) assert
		#endif
		for(v = 0; v < tot_ev_per_dim[1]; v++) //Candidato a vectorizar
		{  	
			//Conversion to PC space			
			PCdot[0] = (eigenvectors->me[0][0] * gridpoint[0]) + (eigenvectors->me[0][1] * (lower[1] + (dx[1] * v)));
		    	PCdot[1] = (eigenvectors->me[1][0] * gridpoint[0]) + (eigenvectors->me[1][1] * (lower[1] + (dx[1] * v)));		    
		
			//Absolute distance calculation
			temp = (((PC[0] - PCdot[0]) * (PC[0] - PCdot[0])) + ((PC[1] - PCdot[1]) * (PC[1] - PCdot[1])) ) / h2;
					
			//If OpenMP version, store the density value in an auxiliar vector densValues, previous to storing in the final PDF structure
			//Vector densPosition will contain the position of the gridpoint in the final PDF structure
			#ifdef _OPENMP

			//PDFposition			
			densPosition[v] = HalfPosition + ((((lower[1] + (dx[1] * v)) - x0[1])/ dx[1]) * pdf->pdfcumsize[1]);		  
			densValues[v] = (0.5/cd*(2+2.)*(1.-temp)) * (fabs(temp)<1.);

			//If serial version, store the density value of the sample over the gridpoint in the PDF structure
			#else

			gridpoint[1] = (lower[1] + (dx[1] * v));

	        	dif_pos[0] = (gridpoint[0] - x0[0])/ dx[0];
			dif_pos[1] = (gridpoint[1] - x0[1])/ dx[1];

			*PDFitem(pdf ,dif_pos, 2) += (0.5/cd*(2+2.)*(1.-temp)) * (fabs(temp)<1.) ;	
			
			#endif	
		}

		#ifdef _OPENMP

		for(v = 0; v < tot_ev_per_dim[1]; v++)
			#pragma omp atomic
			pdf->PDF[densPosition[v]] += densValues[v];
	
		#endif
	}
}

void compute2DBox_3D(PDFPtr pdf, double * PC, double * lower, int * tot_ev_per_dim, double * gridpoint,	size_t * dif_pos, 
	double * x0,double * dx, double h2, double cd, MAT * eigenvectors, double *  densValues, int *  densPosition)
{
	int u,v,l; //Loop variables
	double temp; //Will contain the absolute distance value from gridpoint to sample.
	double PCdot[3] __attribute__((aligned(64)));
	
	//Compiler flag to inform about structure alignment		
	//__assume_aligned(densValues,64);
	//__assume_aligned(densPosition,64);				

	for(gridpoint[0] = lower[0], u = 0; u <= tot_ev_per_dim[0]; gridpoint[0] += dx[0], u++)
	{  
		int HalfPosition = (((gridpoint[0] - x0[0])/ dx[0]) * pdf->pdfcumsize[0]) + (((gridpoint[2] - x0[2])/ dx[2]) * pdf->pdfcumsize[2]);
		
		#pragma simd private(PCdot) assert	
		for(v = 0; v <= tot_ev_per_dim[1]; v++) //Candidato a vectorizar
		{  	
			//Conversion to PC space			
		    	PCdot[0] = (eigenvectors->me[0][0] * gridpoint[0]) + (eigenvectors->me[0][1] * (lower[1] + (dx[1] * v))) + (eigenvectors->me[0][2] * gridpoint[2]);
		    	PCdot[1] = (eigenvectors->me[1][0] * gridpoint[0]) + (eigenvectors->me[1][1] * (lower[1] + (dx[1] * v))) + (eigenvectors->me[1][2] * gridpoint[2]);		    
		    	PCdot[2] = (eigenvectors->me[2][0] * gridpoint[0]) + (eigenvectors->me[2][1] * (lower[1] + (dx[1] * v))) + (eigenvectors->me[2][2] * gridpoint[2]);		
		
			//Absolute distance calculation
			temp = (((PC[0] - PCdot[0]) * (PC[0] - PCdot[0])) + ((PC[1] - PCdot[1]) * (PC[1] - PCdot[1])) + ((PC[2] - PCdot[2]) * (PC[2] - PCdot[2]))) / h2;
					
			//If OpenMP version, store the density value in an auxiliar vector densValues, previous to storing in the final PDF structure
			//Vector densPosition will contain the position of the gridpoint in the final PDF structure
			#ifdef _OPENMP

			//PDFposition			
			densPosition[v] = HalfPosition + ((((lower[1] + (dx[1] * v)) - x0[1])/ dx[1]) * pdf->pdfcumsize[1]);		  
			densValues[v] = (0.5/cd*(3+2.)*(1.-temp)) * (fabs(temp)<1.);

			//If serial version, store the density value of the sample over the gridpoint in the PDF structure
			#else

			gridpoint[1] = (lower[1] + (dx[1] * v));

		        dif_pos[0] = (gridpoint[0] - x0[0])/ dx[0];
			dif_pos[1] = (gridpoint[1] - x0[1])/ dx[1];
			dif_pos[2] = (gridpoint[2] - x0[2])/ dx[2];

			*PDFitem(pdf ,dif_pos, 3) += (0.5/cd*(3+2.)*(1.-temp)) * (fabs(temp)<1.) ;	
			
			#endif	

		}

		#ifdef _OPENMP

		for(v = 0; v <= tot_ev_per_dim[1]; v++)
			#pragma omp atomic
			pdf->PDF[densPosition[v]] += densValues[v];
	
		#endif
	}
}

//Compute the density in the bounding box of a sample - Generic function, used for spaces of dimensionality higher than 3
void compute2DBox_ND(PDFPtr pdf, double * PC, double * lower, int * tot_ev_per_dim, double * gridpoint, size_t * dif_pos, double * x0, 
	double * dx, int dim, double h2, double cd, MAT * eigenvectors, double *  densValues, int *  densPosition, 
	double *  PCdot_vec, double *  temp_vec, double *  gridpoint_vec)
{
	int u,v,d,l; //Loop variables
	
	int HalfPosition;
	int dimGreaterThanTwoPosition = 0;
	double HalfTemp = 0;

	#ifdef _OPENMP //Initializations for vector implementation

	#pragma simd reduction(+:dimGreaterThanTwoPosition) assert
	for(d = 2; d < dim; d++)
		dimGreaterThanTwoPosition += (dif_pos[d] * pdf->pdfcumsize[d]);
		
	for(v = 0; v < tot_ev_per_dim[1]; v++) 
		for(d = 2; d < dim; d++)
			gridpoint_vec[v * dim + d] = gridpoint[d];
	#endif
	
	for(gridpoint[0] = lower[0], u = 0; u < tot_ev_per_dim[0]; gridpoint[0] += dx[0], u++)
	{  		
		//Compiler flag to inform about structure alignment		
		//__assume_aligned(densValues,64);
		//__assume_aligned(densPosition,64);		
		
		#ifdef _OPENMP //Vector friendly implementation
		
		HalfPosition = (((gridpoint[0] - x0[0])/ dx[0]) * pdf->pdfcumsize[0]) + dimGreaterThanTwoPosition;		
				
		for(v = 0; v < tot_ev_per_dim[1]; v++) 
			gridpoint_vec[v * dim + 0] = gridpoint[0];
			
		for(v = 0; v < tot_ev_per_dim[1]; v++) 	
			temp_vec[v] = 0;
		
		for(v = 0; v < tot_ev_per_dim[1]; v++) 
			gridpoint_vec[v * dim + 1] = (lower[1] + (dx[1] * v));
		
		for(v = 0; v < tot_ev_per_dim[1] * dim; v++) 
			PCdot_vec[v] = 0;	
				
		for(v = 0; v < tot_ev_per_dim[1]; v++) 
			for(d = 0; d < dim; d++)	
				#pragma simd reduction(+:PCdot_vec[v * dim + d]) assert
				for(l = 0; l < dim; l++)
					PCdot_vec[v * dim + d] += (eigenvectors->me[d][l] * gridpoint_vec[v * dim + l]);		
						
		for(v = 0; v < tot_ev_per_dim[1]; v++) 			
			#pragma simd reduction(+:temp_vec[v]) assert
			for(d = 0; d < dim; d++)				
				temp_vec[v] += ((PC[d] - PCdot_vec[v * dim + d]) * (PC[d] - PCdot_vec[v * dim + d]));
		
		for(v = 0; v < tot_ev_per_dim[1]; v++) 		
			temp_vec[v] /= h2;
			
		for(v = 0; v < tot_ev_per_dim[1]; v++) 
			densPosition[v] = HalfPosition + ((((lower[1] + (dx[1] * v)) - x0[1])/ dx[1]) * pdf->pdfcumsize[1]);		  
		
		for(v = 0; v < tot_ev_per_dim[1]; v++) 
			densValues[v] = (0.5/cd*(dim + 2.)*(1.-temp_vec[v])) * (fabs(temp_vec[v])<1.);
			
		for(v = 0; v < tot_ev_per_dim[1]; v++)
			#pragma omp atomic
			pdf->PDF[densPosition[v]] += densValues[v];		
		
		#else	// Serial implementation
		
		double temp;		
		dif_pos[0] = (gridpoint[0] - x0[0])/ dx[0];		
		
		for(v = 0; v < tot_ev_per_dim[1]; v++) 
		{  						
			gridpoint[1] = (lower[1] + (dx[1] * v));
			
			//Conversion to PC space						
			for(d = 0; d < dim; d++)
				PCdot_vec[d] = 0;	
			
			for(d = 0; d < dim; d++)	
				#pragma simd reduction(+:PCdot_vec[d]) assert
				for(l = 0; l < dim; l++)
					PCdot_vec[d] += (eigenvectors->me[d][l] * gridpoint[l]);
				
			//Absolute distance calculation	
			temp = 0;
			
			#pragma simd reduction(+:temp) assert
			for(d = 0; d < dim; d++)
				temp += ((PC[d] - PCdot_vec[d]) * (PC[d] - PCdot_vec[d]));
				
			temp /= h2;	
			
 			dif_pos[1] = (gridpoint[1] - x0[1])/ dx[1];

			*PDFitem(pdf ,dif_pos, dim) += (0.5/cd*(dim + 2.)*(1.-temp)) * (fabs(temp)<1.) ;				
		}		
		
		#endif		
	}
}

/**** Functions to calculate PDF, called from main ****/

//Compute the PDF of a one-dimensional grid space
int computePDF1D(MPDFEstimatorPtr mpdf, PDFPtr pdf, MAT *Sm1 , double h , double detSm1 , double *x0, 
		double *x1, double *dx, double *bounds, MAT *eigenvectors )
{
  int i,j,u; //Loop variables
  int dim = 1;	//Dimensions of grid space
  double cd = volumeConstant(dim); //Volume constants to calculate kernel values    
  double h2=h*h;  //Squared bandwith value
  double *PC; // Current sample (PC space)
  double theintegral = 0.0;  
  double total_vol = 0.0;  
  double * sample; 
  double k=1./sqrt(detSm1)/mpdf->current/pow(h,mpdf->length);  //Constant to recover the volume in the X space from the volume in the PC space
  double PCdot;
   
 //Variables to calculate coordinates and number of gridpoints of bounding box
  int steps;	  
  double upper, lower, gridpoint;
  int tot_ev;
  size_t dif_pos[1];
  double abs_bound,temp;   
  
  //Auxiliary vectors for OpenMP version
  double * densValues;
  int * densPosition;    
        
  #pragma omp parallel default(none) \
  shared(stdout,mpdf,pdf,dim,x0,x1,dx,theintegral,total_vol,bounds,eigenvectors,cd,h2,k) \
  private(i,j,u,sample,PC,lower,upper,steps,abs_bound,tot_ev,dif_pos,gridpoint,PCdot,densValues,densPosition,temp) 
  {	
 
  #ifdef _OPENMP 

  int dim0_max_size = ((ceil(bounds[0] / dx[0]) * 2) + 3);

  densValues = (double *)malloc(sizeof(double) * dim0_max_size); //Vector to hold density values of each sample-gridpoint combination
  densPosition = (int *)malloc(sizeof(int) * dim0_max_size); //Vector to hold the positions of densValues values in the PDF structure

  #endif     
  
  //Initialize PDF structure to 0s
  #pragma omp for
  for(i = 0; i < pdf->total_size; i++)
	pdf->PDF[i] = 0.0f;  
  
  //Main calculation loop. For each sample calculate the PDF of its influence area and store in the PDF structure
  #pragma omp for
  for(i=0;i<mpdf->current;i++) 
  {	
	sample = MPDFPosition(mpdf,i); //Get current sample
	PC = MPDFPCPosition(mpdf,i); //Get current sample (scaled as PC)
	  	  
	//For each sample, calculate its boundaries
	
	//Lower corner
	abs_bound = sample[0] - bounds[0];
	if (x0[0] > abs_bound)
		lower = x0[0];
	else
	{
		steps = floor((abs_bound - x0[0]) / dx[0]);
		lower = x0[0] + (steps * dx[0]);
	}

	//Upper corner
	abs_bound = sample[0] + bounds[0];
	if (x1[0] < abs_bound)
		upper = x1[0];
	else
	{
		steps = ceil((abs_bound - x0[0]) / dx[0]);
		upper = x0[0] + (steps * dx[0]);
	}	
	
	//Calculate number of eval points per dimension	
	tot_ev = rint((upper - lower)/dx[0]) + 1;		   
  

	//Calculate the PDF of the defined 1D space
	#ifdef _OPENMP
	#pragma simd private(PCdot,temp) assert
	#endif
	for(u = 0; u < tot_ev; u++)
	{  		
	    PCdot = (eigenvectors->me[0][0] * (lower + (dx[0] * u)));
	
		//Absolute distance calculation
		temp = ((PC[0] - PCdot) * (PC[0] - PCdot)) / h2;

		//If OpenMP version, store the density value in an auxiliar vector densValues, previous to storing in the final PDF structure
		//Vector densPosition will contain the position of the gridpoint in the final PDF structure
		#ifdef _OPENMP

		//PDFposition			
		densPosition[u] = (((lower + (dx[0] * u)) - x0[0])/ dx[0]) * pdf->pdfcumsize[0];		  
		densValues[u] = (0.5/cd*(1+2.)*(1.-temp)) * (fabs(temp)<1.);

		//If serial version, store the density value of the sample over the gridpoint in the PDF structure
		#else

		dif_pos[0] = ((lower + (dx[0] * u)) - x0[0])/ dx[0];
		*PDFitem(pdf ,dif_pos, 1) += (0.5/cd*(1+2.)*(1.-temp)) * (fabs(temp)<1.) ;	
		
		#endif			
	}	
	
	#ifdef _OPENMP

	for(u = 0; u < tot_ev; u++)
		#pragma omp atomic
		pdf->PDF[densPosition[u]] += densValues[u];

	#endif	    	 
  } 
    
  #ifdef _OPENMP
  free(densValues);
  free(densPosition);      
  #endif

  //Apply k constant to PDF
  #pragma omp for
  for(i=0; i < pdf->total_size; i++)
  	  pdf->PDF[i] = pdf->PDF[i] * k;

  //Calculate integral of PDF
  #pragma omp for reduction(+:theintegral)  
  for(i=0; i < pdf->total_size; i++)
      theintegral += pdf->PDF[i];
      
  #pragma omp single
  theintegral = theintegral * dx[0];     

  //Renormalize PDF using integral
  #pragma omp for
  for(i=0; i < pdf->total_size; i++)
     pdf->PDF[i] = pdf->PDF[i]/theintegral;
     
  //Calculate total volume of renormalized PDF   
  #pragma omp for reduction(+:total_vol)  
  for(i=0; i < pdf->total_size; i++)
	total_vol += pdf->PDF[i];
  
  }//End of parallel OpenMP Region    
  
  printf("Total integrated PDF: %g. The integral: %f\n",total_vol*dx[0],theintegral);	   

  return 0;
} 

//Compute the PDF of a 2D grid space
int computePDF2D(MPDFEstimatorPtr mpdf, PDFPtr pdf, MAT *Sm1 , double h , double detSm1 , double *x0, 
		double *x1, double *dx, double *bounds, MAT *eigenvectors, oclVars * opencl )
{
  int i,u; //Loop variables
  int dim = 2;	//Dimensions of grid space
  double cd = volumeConstant(dim); //Volume constants to calculate kernel values    
  double h2=h*h;  //Squared bandwith value
  double theintegral = 0.0;  
  double total_vol = 0.0;  
  double total_dx = dx[0] * dx[1]; 
  double k=1./sqrt(detSm1)/mpdf->current/pow(h,mpdf->length);  //Constant to recover the volume in the X space from the volume in the PC space

  size_t local_item_size; //Workgroup size
  size_t global_item_size; // ND Range size		    
  int groupAmount; 
  int status;

  //Calculate initial params: padded number of samples, ...
  int nsamples_pad1 = mpdf->current + 1;
  int nsamples_pad4;

  if((nsamples_pad1 % 4) == 0)
	nsamples_pad4 = nsamples_pad1;
  else
	nsamples_pad4 = ((nsamples_pad1 / 4) + 1) * 4;

  int max_gp_sample = ((bounds[0] * 2) / dx[0]) + 1 * (((bounds[1] * 2) / dx[1]) + 1);
  int chunk_size = 1024;

  double * density_values = (double *)malloc(sizeof(double) * chunk_size * max_gp_sample);
  int * density_positions = (int *)malloc(sizeof(int) * chunk_size * max_gp_sample);
  
  //Create initial OpenCL buffers
  status = createInitialBuffers(opencl, dim, mpdf, pdf, NULL, eigenvectors, bounds, x0, x1, dx, pdf->pdfcumsize, NULL, nsamples_pad4, chunk_size, 1, max_gp_sample);
  if(status != CL_SUCCESS) {printf("\nError createInitialBuffers"); return 1;}  
  
  //Set kernel args
  status = setKernelArgs(opencl, mpdf->current, nsamples_pad4, h2, cd, dim);
  if(status != CL_SUCCESS) {printf("\nError setKernelArgs"); return 1;}   

  //Initialize PDF structure to 0s  
  #pragma omp parallel for
  for(i = 0; i < pdf->total_size; i++)
	pdf->PDF[i] = 0.0f;

  ////////// Launch kernel to calculate the bounds of each sample

  local_item_size = opencl->wg_size;
  groupAmount = ceil_int(mpdf->current,opencl->wg_size);
  global_item_size = local_item_size * groupAmount;
  
  status = clEnqueueNDRangeKernel(opencl->command_queue, opencl->kernel_calc_gp_2D, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
  if (status != CL_SUCCESS) {	printf("\nError: clEnqueueNDRangeKernel kernel_samples_bounds(): %d ",status);	return 1; }	          
 
  ////////// Scan number of gridpoints

  // For scan, we use a reduce-then-scan approach

  // Each thread block gets an equal portion of the input array, and computes the sum.
  status = clEnqueueNDRangeKernel(opencl->command_queue, opencl->kernel_reduce, 1, NULL, &opencl->scan_global_wsize, &opencl->scan_local_wsize, 0, NULL, NULL);
  if (status != CL_SUCCESS) {	printf("\nError: clEnqueueNDRangeKernel kernel_reduce: %d ",status);	return 1; }	          

  // Next, a top-level exclusive scan is performed on the array of block sums
  status = clEnqueueNDRangeKernel(opencl->command_queue, opencl->kernel_top_scan, 1, NULL,&opencl->scan_local_wsize, &opencl->scan_local_wsize, 0, NULL, NULL);
  if (status != CL_SUCCESS) {	printf("\nError: clEnqueueNDRangeKernel kernel_top_scan: %d ",status);	return 1; }	          

  // Finally, a bottom-level scan is performed by each block
  // that is seeded with the scanned value in block sums
  status = clEnqueueNDRangeKernel(opencl->command_queue, opencl->kernel_bottom_scan, 1, NULL,&opencl->scan_global_wsize, &opencl->scan_local_wsize, 0, NULL, NULL);
  if (status != CL_SUCCESS) {	printf("\nError: clEnqueueNDRangeKernel kernel_bottom_scan: %d ",status);	return 1; }	          

  for(i = 0; i < mpdf->current; i += chunk_size)
  {
	////////// Launch kernel to calculate densities
	
	int first = i;
	int last = ((i + chunk_size) < mpdf->current) ? (i + chunk_size) : mpdf->current;
	int chunk_samples = last - first;

	local_item_size = opencl->wg_size;
  	groupAmount = ceil_int(chunk_samples,opencl->wg_size);
 	global_item_size = local_item_size * groupAmount;
 
    	status  = clSetKernelArg(opencl->kernel_density_2D, 0, sizeof(cl_int), (void*)&first);
    	status |= clSetKernelArg(opencl->kernel_density_2D, 1, sizeof(cl_int), (void*)&chunk_samples);
 	if (status != CL_SUCCESS) {	printf("\nError: setKernelArg density 2D: %d ",status);	return 1; }	          
 
  	status = clEnqueueNDRangeKernel(opencl->command_queue, opencl->kernel_density_2D, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
  	if (status != CL_SUCCESS) {	printf("\nError: clEnqueueNDRangeKernel kernel_samples_bounds(): %d ",status);	return 1; }	          
 
  	////////// Retrieve gridpoints and accumulate

	int chunk_gp_number;

	status = clEnqueueReadBuffer(opencl->command_queue, opencl->gp_xy_scan, CL_TRUE, sizeof(int) * last, sizeof(int), &chunk_gp_number, 0, NULL, NULL);
	if (status != CL_SUCCESS) { printf("\nError: clEnqueueReadBuffer (chunk_gp_number): %d Sample: %d \n",status,i); return 1; }      

	status = clEnqueueReadBuffer(opencl->command_queue, opencl->density_positions, CL_TRUE, 0, sizeof(int) * chunk_gp_number, density_positions, 0, NULL, NULL);
	if (status != CL_SUCCESS) { printf("\nError: clEnqueueReadBuffer (temp_dbuffer positions): %d Sample: %d total_eval_points_chunk %d \n",status,i,chunk_gp_number); return 1; }      

	status = clEnqueueReadBuffer(opencl->command_queue, opencl->density_values, CL_TRUE, 0, sizeof(double) * chunk_gp_number, density_values, 0, NULL, NULL);
	if (status != CL_SUCCESS) { printf("\nError: clEnqueueReadBuffer (temp_dbuffer values): %d Sample: %d total_eval_points_chunk %d\n",status,i,chunk_gp_number); return 1; }    
	
	for(u = 0; u < chunk_gp_number; u++)
		pdf->PDF[density_positions[u]] += density_values[u];		
  }

  free(density_positions);
  free(density_values);

  ////////// Final calculations
             
  #pragma omp parallel default(none) shared(pdf,k,total_dx,theintegral,total_vol) private(i) 
  {	
 
  //Apply k constant to PDF
  #pragma omp for
  for(i=0; i < pdf->total_size; i++)
  	  pdf->PDF[i] = pdf->PDF[i] * k;

  //Calculate integral of PDF
  #pragma omp for reduction(+:theintegral)   
  for(i=0; i < pdf->total_size; i++)
      theintegral += pdf->PDF[i];

  #pragma omp single
  theintegral = theintegral * total_dx;

  //Renormalize PDF using integral
  #pragma omp for
  for(i=0; i < pdf->total_size; i++)
     pdf->PDF[i] = pdf->PDF[i]/theintegral;
     
  //Calculate total volume of renormalized PDF   
  #pragma omp for reduction(+:total_vol)  
  for(i=0; i < pdf->total_size; i++)
	total_vol += pdf->PDF[i];
  
  }//End of parallel OpenMP Region    
  
  printf("Total integrated PDF: %g. The integral: %f\n",total_vol*dx[0]*dx[1],theintegral);

  return 0;	
}

#define DEBUG_TEMPS 1
#undef  DEBUG_TEMPS

//Compute the PDF of grid spaces of dimension 3 or higher
int computePDF3D(MPDFEstimatorPtr mpdf, PDFPtr pdf, MAT *Sm1 , double h , double detSm1 , double *x0, double *x1, double *dx, 
		  double *bounds, MAT *eigenvectors, oclVars * opencl)
{
  int dim = 3; 
  int i,j,u; //Loop variables	
  double cd = volumeConstant(dim); //Volume constant
  double k=1./sqrt(detSm1)/mpdf->current/pow(h,mpdf->length); //Constant to recover the volume in the X space from the volume in the PC space  
  double h2=h*h; //Square of bandwith value
  double total_vol=0.0;
  double theintegral=0.0;

   //Calculate acumulated volume for the grid space
  double total_dx = 1.0;
  for (i = 0; i < dim; i++)
	total_dx *= dx[i];   

  //Apply BW effect to Sm1  
  for(i = 0; i < 3; i++) 
	for(j = 0; j < 3; j++) 
		Sm1->me[i][j] /= h2; 

  //Calculate partial equations for the 2D layering    							
  double A = Sm1->me[0][0];
  double B = 2 * Sm1->me[0][1];
  double C = Sm1->me[1][1];
  double theta = atan(B/(A-C))/2;		 
  double cosTheta = cos(theta);
  double sinTheta = sin(theta);					
  double X2 = Sm1->me[0][0]*cosTheta*cosTheta + 2*Sm1->me[0][1]*cosTheta*sinTheta +   Sm1->me[1][1]*sinTheta*sinTheta;
  double Y2 = Sm1->me[0][0]*sinTheta*sinTheta - 2*Sm1->me[0][1]*cosTheta*sinTheta +   Sm1->me[1][1]*cosTheta*cosTheta;		
  
  //Create OpenCL structures
  croppingVars cp;  
  cp.cosTheta = cosTheta;
  cp.sinTheta = sinTheta;
  cp.X2 = X2;
  cp.Y2 = Y2;    
  int groupAmount;
  size_t local_item_size[2]; //Workgroup size
  size_t global_item_size[2]; // ND Range size		    
  int status;

  //Calculate initial params: padded number of samples, ...
  int nsamples_pad1 = mpdf->current + 1;
  int nsamples_pad4;

  if((nsamples_pad1 % 4) == 0)
	nsamples_pad4 = nsamples_pad1;
  else
	nsamples_pad4 = ((nsamples_pad1 / 4) + 1) * 4;

  //int chunk_size = 2048 * 8; //Number of samples to be processed per chunk of work 
  int chunk_size = 256; //Number of samples to be processed per chunk of work 

  int max_slices_sample = ((bounds[2] * 2) / dx[2]) + 1;

  int max_gp_sample = ((bounds[0] * 2) / dx[0]) + 1;
  for(i = 1; i < dim; i++)
	max_gp_sample *= (((bounds[i] * 2) / dx[i]) + 1);

  //Create initial OpenCL buffers
  status = createInitialBuffers(opencl, dim, mpdf, pdf, Sm1, eigenvectors, bounds, x0, x1, dx, pdf->pdfcumsize, &cp, nsamples_pad4, chunk_size, max_slices_sample, max_gp_sample);
  if(status != CL_SUCCESS) {printf("\nError createInitialBuffers"); return 1;}  
  
  //Set kernel args
  status = setKernelArgs(opencl, mpdf->current, nsamples_pad4, h2, cd, dim);
  if(status != CL_SUCCESS) {printf("\nError setKernelArgs"); return 1;}  
  
  ////////// Launch kernel to calculate the bounds in first two dimensions for every sample

  local_item_size[0] = opencl->wg_size;
  groupAmount = ceil_int(mpdf->current,opencl->wg_size);
  global_item_size[0] = local_item_size[0] * groupAmount;
  
  status = clEnqueueNDRangeKernel(opencl->command_queue, opencl->kernel_calc_slices, 1, NULL, global_item_size, local_item_size, 0, NULL, NULL);			
  if (status != CL_SUCCESS) {	printf("\nError: clEnqueueNDRangeKernel kernel_samples_bounds(): %d ",status);	return 1; }	          
 
  ////////// Scan slices

  // For scan, we use a reduce-then-scan approach

  // Each thread block gets an equal portion of the input array, and computes the sum.
  status = clEnqueueNDRangeKernel(opencl->command_queue, opencl->kernel_reduce, 1, NULL, &opencl->scan_global_wsize, &opencl->scan_local_wsize, 0, NULL, NULL);
  if (status != CL_SUCCESS) {	printf("\nError: clEnqueueNDRangeKernel kernel_reduce: %d ",status);	return 1; }	          

  // Next, a top-level exclusive scan is performed on the array of block sums
  status = clEnqueueNDRangeKernel(opencl->command_queue, opencl->kernel_top_scan, 1, NULL,&opencl->scan_local_wsize, &opencl->scan_local_wsize, 0, NULL, NULL);
  if (status != CL_SUCCESS) {	printf("\nError: clEnqueueNDRangeKernel kernel_top_scan: %d ",status);	return 1; }	          

  // Finally, a bottom-level scan is performed by each block
  // that is seeded with the scanned value in block sums
  status = clEnqueueNDRangeKernel(opencl->command_queue, opencl->kernel_bottom_scan, 1, NULL,&opencl->scan_global_wsize, &opencl->scan_local_wsize, 0, NULL, NULL);
  if (status != CL_SUCCESS) {	printf("\nError: clEnqueueNDRangeKernel kernel_bottom_scan: %d ",status);	return 1; }	          

  ////////// Fetch scanned vector and set params

  double * density_values = (double *)malloc(sizeof(double) * chunk_size * max_gp_sample);
  int * density_positions = (int *)malloc(sizeof(int) * chunk_size * max_gp_sample);
  int * slices_scan = (int *)malloc(sizeof(int) * (mpdf->current + 1));

  status = clEnqueueReadBuffer(opencl->command_queue, opencl->slices_sample_scan, CL_TRUE, 0, (mpdf->current + 1) * sizeof(int), slices_scan, 0, NULL, NULL);
  if (status != CL_SUCCESS) {	printf("\nError: ReadBufer Slices_scan: %d ",status);	return 1; }	          
	
  //Initialize PDF structure to 0s
  #pragma omp parallel for 
  for(i = 0; i < pdf->total_size; i++)
	pdf->PDF[i] = 0.0f;

  //Switch args for scan kernels to gp_xy vector
  status  = clSetKernelArg(opencl->kernel_reduce, 0, sizeof(cl_mem), (void*)&opencl->gp_xy);
  status |= clSetKernelArg(opencl->kernel_bottom_scan, 0, sizeof(cl_mem), (void*)&opencl->gp_xy);
  status |= clSetKernelArg(opencl->kernel_bottom_scan, 2, sizeof(cl_mem), (void*)&opencl->gp_xy_scan);
  if (status != CL_SUCCESS) {	printf("\nError: setKernelArgs for scan gp number: %d ",status);	return 1; }	          

  int padding[3] = {0,0,0};
	
  for(i = 0; i < mpdf->current; i += chunk_size)
  {
	int first = i;
	int last = ((i + chunk_size) < mpdf->current) ? (i + chunk_size) : mpdf->current;
	int chunk_samples = last - first;


	//1st: Calculate number of gridpoints per sample
	local_item_size[0] = opencl->wg_size;
  	groupAmount = ceil_int(chunk_samples,opencl->wg_size);
 	global_item_size[0] = local_item_size[0] * groupAmount;
  
	local_item_size[1] = 1;
	global_item_size[1] = max_slices_sample;

    	status  = clSetKernelArg(opencl->kernel_calc_gp, 0, sizeof(cl_int), (void*)&first);
 	if (status != CL_SUCCESS) {	printf("\nError: setKernelArg calc offset: %d ",status);	return 1; }	          

	status = clEnqueueNDRangeKernel(opencl->command_queue, opencl->kernel_calc_gp, 2, NULL, global_item_size, local_item_size, 0, NULL, NULL);
	if (status != CL_SUCCESS) {	printf("\nError: clEnqueueNDRangeKernel kernel_calc_gp: %d ",status);	return 1; }	          


	//2nd: Scan number of gridpoints for indexed storage

	int chunk_slices = slices_scan[last] - slices_scan[first] + 1; //Sum of the slices per sample in this chunk
	int scan_items = chunk_slices + 1; //Number of items to process in scan, +1 is to store final reduction in last position

	status = clEnqueueWriteBuffer(opencl->command_queue, opencl->gp_xy, CL_TRUE, sizeof(int) * chunk_slices , sizeof(int), padding, 0, NULL, NULL);
	if (status != CL_SUCCESS) {	printf("\nError: clEnqueueWriteBuffer zero in gp_number_scan: %d ",status);	return 1; }	          

	//Pad scan_items to multiple of 4
	if(((scan_items) % 4) > 0)
	{
		int nslices_pad4 = ((scan_items / 4) + 1) * 4;
		int pad_number = nslices_pad4 - scan_items;

		status = clEnqueueWriteBuffer(opencl->command_queue, opencl->gp_xy, CL_TRUE, sizeof(int) * scan_items, sizeof(int) * pad_number, padding, 0, NULL, NULL);
		if (status != CL_SUCCESS) {	printf("\nError: clEnqueueWriteBuffer zero in gp_number_scan: %d ",status);	return 1; }	          
		scan_items += pad_number;
	}

   	status  = clSetKernelArg(opencl->kernel_reduce, 2, sizeof(cl_int), (void*)&scan_items);
	status |= clSetKernelArg(opencl->kernel_bottom_scan, 3, sizeof(cl_int), (void*)&scan_items);
	if (status != CL_SUCCESS) {	printf("\nError: clEnqueueNDRangeKernel kernel_reduce: %d ",status);	return 1; }	          

	// For scan, we use a reduce-then-scan approach

	status = clEnqueueNDRangeKernel(opencl->command_queue, opencl->kernel_reduce, 1, NULL, &opencl->scan_global_wsize, &opencl->scan_local_wsize, 0, NULL, NULL);
	if (status != CL_SUCCESS) {	printf("\nError: clEnqueueNDRangeKernel kernel_reduce: %d ",status);	return 1; }	          

	status = clEnqueueNDRangeKernel(opencl->command_queue, opencl->kernel_top_scan, 1, NULL,&opencl->scan_local_wsize, &opencl->scan_local_wsize, 0, NULL, NULL);
	if (status != CL_SUCCESS) {	printf("\nError: clEnqueueNDRangeKernel kernel_top_scan: %d ",status);	return 1; }	          

	status = clEnqueueNDRangeKernel(opencl->command_queue, opencl->kernel_bottom_scan, 1, NULL,&opencl->scan_global_wsize, &opencl->scan_local_wsize, 0, NULL, NULL);
	if (status != CL_SUCCESS) {	printf("\nError: clEnqueueNDRangeKernel kernel_bottom_scan: %d ",status);	return 1; }	          


/*	 if(first == 12032)
	 {
	 //DEBUG
	 printf("\nfirst %d last %d chunk_samples %d slices in chunk: first %d last %d total %d",first,last,chunk_samples,slices_scan[first],slices_scan[last ],chunk_slices);
	fflush(stdout);

	 clFinish(opencl->command_queue);
	
	 int size = chunk_slices + 1;
	 int * debug = (int *)malloc(sizeof(int) * size); 
	 int * debug2 = (int *)malloc(sizeof(int) * size); 

	 status = clEnqueueReadBuffer(opencl->command_queue, opencl->gp_xy, CL_TRUE, 0, size * sizeof(int), debug, 0, NULL, NULL);
	 if (status != CL_SUCCESS) {	printf("\nError: WriteBuffer debug): %d ",status);	return 1; }	          
	 
	 status = clEnqueueReadBuffer(opencl->command_queue, opencl->gp_xy_scan, CL_TRUE, 0, size * sizeof(int), debug2, 0, NULL, NULL);
	 if (status != CL_SUCCESS) {	printf("\nError: WriteBuffer debug): %d ",status);	return 1; }	          

	 printf("\nDEBUG (%d): ",size);
	for(i = 0; i < size; i++)
		printf("\n%3d: %5d, %5d",i,debug[i],debug2[i]);

	fflush(stdout);
	free(debug);
	free(debug2);
	//return 1;
	}*/


	//3rd: Calculate densities
	local_item_size[0] = opencl->wg_size;
  	groupAmount = ceil_int(chunk_slices,opencl->wg_size);
 	global_item_size[0] = local_item_size[0] * groupAmount;

   	status = clSetKernelArg(opencl->kernel_density, 0, sizeof(cl_int), (void*)&chunk_slices);
 	if (status != CL_SUCCESS) {	printf("\nError: setKernelArg density: %d ",status);	return 1; }	          
  
	status = clEnqueueNDRangeKernel(opencl->command_queue, opencl->kernel_density, 1, NULL, global_item_size, local_item_size, 0, NULL, NULL);
	if (status != CL_SUCCESS) {	printf("\nError: clEnqueueNDRangeKernel kernel_calc_gp: %d ",status);	return 1; }	          


	//4th: Retrieve Results
	int chunk_gp_number;

	status = clEnqueueReadBuffer(opencl->command_queue, opencl->gp_xy_scan, CL_TRUE, sizeof(int) * chunk_slices, sizeof(int), &chunk_gp_number, 0, NULL, NULL);
	if (status != CL_SUCCESS) { printf("\nError: clEnqueueReadBuffer (chunk_gp_number): %d Sample: %d \n",status,i); return 1; }      

	status = clEnqueueReadBuffer(opencl->command_queue, opencl->density_positions, CL_TRUE, 0, sizeof(int) * chunk_gp_number, density_positions, 0, NULL, NULL);
	if (status != CL_SUCCESS) { printf("\nError: clEnqueueReadBuffer (temp_dbuffer positions): %d Sample: %d total_eval_points_chunk %d \n",status,i,chunk_gp_number); return 1; }      

	status = clEnqueueReadBuffer(opencl->command_queue, opencl->density_values, CL_TRUE, 0, sizeof(double) * chunk_gp_number, density_values, 0, NULL, NULL);
	if (status != CL_SUCCESS) { printf("\nError: clEnqueueReadBuffer (temp_dbuffer values): %d Sample: %d total_eval_points_chunk %d\n",status,i,chunk_gp_number); return 1; }    
	
	for(u = 0; u < chunk_gp_number; u++)
		pdf->PDF[density_positions[u]] += density_values[u];		
  }

  free(density_values);
  free(density_positions);
  free(slices_scan);

  //Beginning of OpenMP parallel region								
  #pragma omp parallel default(none) shared(pdf,k,theintegral,total_vol,total_dx) 
  {			
								
  //Apply k constant to PDF
  #pragma omp for
  for(i=0; i < pdf->total_size; i++)
  	  pdf->PDF[i] = pdf->PDF[i] * k;

  //Calculate integral of PDF  
  #pragma omp for reduction(+:theintegral) 
  for(i=0; i < pdf->total_size; i++)
      theintegral += pdf->PDF[i];

  #pragma omp single
  theintegral = theintegral * total_dx;
  
  //Renormalize PDF using integral
  #pragma omp for
  for(i=0; i < pdf->total_size; i++)
     pdf->PDF[i] = pdf->PDF[i]/theintegral;
  
  //Calculate total volume of renormalized PDF     
  #pragma omp for reduction(+:total_vol)  
  for(i=0; i < pdf->total_size; i++)
	total_vol += pdf->PDF[i];

  }//End of parallel OpenMP Region   
  
  
  printf("Total integrated PDF: %g. The integral: %f\n",total_vol*total_dx,theintegral);

  return 0;
}

//Compute the PDF of grid spaces of dimension 3 or higher
int computePDFND(MPDFEstimatorPtr mpdf, PDFPtr pdf, MAT *Sm1 , double h , double detSm1 , double *x0, double *x1, double *dx, 
		  double *bounds, MAT *eigenvectors, int dim, oclVars * opencl)
{
  int i,j,l,u,w; //Loop variables	
  double cd = volumeConstant(dim); //Volume constant
  double k=1./sqrt(detSm1)/mpdf->current/pow(h,mpdf->length); //Constant to recover the volume in the X space from the volume in the PC space  
  double h2=h*h; //Square of bandwith value
  double *PC; // Current sample (PC space)
  double total_vol=0.0;
  double theintegral=0.0;
  double * sample; //Current sample
  double * PCdot;
  
  //Variables to calculate the bounding box of a sample
  double * lower;
  double upper;
  double * gridpoint;
  int * tot_ev_per_dim;
  size_t * dif_pos;
  int total_ev;	
  int steps;
  double abs_bound; //Absolute bound per sample and dimension, given by ellipsoid shape

  //Calculate acumulated volume for the grid space
  double total_dx = 1.0;
  for (i = 0; i < dim; i++)
	total_dx *= dx[i];   

  //Apply BW effect to Sm1  
  for(i = 0; i < dim; i++) 
	for(j = 0; j < dim; j++) 
		Sm1->me[i][j] /= h2; 

  //Variables to perform the calculation of the 2D layering
  double A,B,C,F,Z,theta,cosTheta,sinTheta,X2,Y2,X,Y,XY,termY2,valor,termX2,upy,rightx,upx_rot,upy_rot,rightx_rot,righty_rot; 
  double bound[2],box_center[2],box_min[2],box_max[2],box_steps[2],box_upper[2];
      			
  //Calculate partial equations for the 2D layering    							
  A = Sm1->me[0][0];
  B = 2 * Sm1->me[0][1];
  C = Sm1->me[1][1];
  theta = atan(B/(A-C))/2;		 
  cosTheta = cos(theta);
  sinTheta = sin(theta);					
  X2 =    Sm1->me[0][0]*cosTheta*cosTheta + 2*Sm1->me[0][1]*cosTheta*sinTheta +   Sm1->me[1][1]*sinTheta*sinTheta;
  XY = -2*Sm1->me[0][0]*cosTheta*sinTheta + 2*Sm1->me[0][1]*cosTheta*cosTheta - 2*Sm1->me[0][1]*sinTheta*sinTheta + 2*Sm1->me[1][1]*cosTheta*sinTheta;
  Y2 =    Sm1->me[0][0]*sinTheta*sinTheta - 2*Sm1->me[0][1]*cosTheta*sinTheta +   Sm1->me[1][1]*cosTheta*cosTheta;		
  
  //Aux vector for OpenMP version
  double * densValues; 
  int * densPosition;
  double * temp_vec;
  double * PCdot_vec;
  double * gridpoint_vec;
 
  //Beginning of OpenMP parallel region								
  #pragma omp parallel default(none)\
  shared(stdout,theintegral,total_vol,total_dx,k,mpdf,pdf,cd,dim,bounds,x0,x1,dx,Sm1,cosTheta,sinTheta,eigenvectors,X2,XY,Y2,h2,h) \
  private(i,j,l,u,w,sample,PC,gridpoint,total_ev,abs_bound,lower,box_upper,tot_ev_per_dim,box_steps,F,X,Y,Z,termX2,termY2,upy,rightx,upx_rot,upy_rot, \
  valor,rightx_rot,righty_rot,bound,box_center,box_min,box_max,PCdot,dif_pos,steps,upper,densValues,densPosition,temp_vec,gridpoint_vec,PCdot_vec)
  {			
								
  //Allocate variables to calculate the bounding box of a sample			
  lower = (double *)malloc(sizeof(double) * dim);
  gridpoint = (double *)malloc(sizeof(double) * dim);
  tot_ev_per_dim = (int *)malloc(sizeof(int) * dim);
  dif_pos = (size_t *)malloc(sizeof(size_t) * dim);			

  #ifdef _OPENMP

  int dim1_max_size = ((ceil(bounds[1] / dx[1]) * 2) + 3);

  densValues = (double *)malloc(sizeof(double) * dim1_max_size);
  densPosition = (int *)malloc(sizeof(int) * dim1_max_size);
  
  temp_vec = (double *)malloc(sizeof(double) * dim1_max_size);
  gridpoint_vec = (double *)malloc(sizeof(double) * dim1_max_size * dim);
  PCdot_vec = (double *)malloc(sizeof(double) * dim1_max_size * dim);
  
  #else
  
  PCdot_vec = (double *)malloc(sizeof(double) * dim);

  #endif

  //Initialize PDF structure to 0s
  #pragma omp for 
  for(i = 0; i < pdf->total_size; i++)
	pdf->PDF[i] = 0.0f;

  //Main calculation loop. For each sample calculate the PDF of its influence area and store in the PDF structure
  #pragma omp for  
  for(i=0;i<mpdf->current;i++) 
  {	  
	sample = MPDFPosition(mpdf,i); //Get current sample
	PC = MPDFPCPosition(mpdf,i); //X is the current sample (scaled as PC)

	//For each sample, calculate its bounding box, 
	//expressed as coordinates of lower corner and number of gridpoints per dimensions
	total_ev = 1;			
	for(j = 2; j < dim; j++)
	{	
		//Lower corner
		abs_bound = sample[j] - bounds[j];
		if (x0[j] > abs_bound)
			lower[j] = x0[j];
		else
		{
			steps = floor((abs_bound - x0[j]) / dx[j]);
			lower[j] = x0[j] + (steps * dx[j]);
		}

		//Upper corner
		abs_bound = sample[j] + bounds[j];
		if (x1[j] < abs_bound)
			upper = x1[j];
		else
		{
			steps = ceil((abs_bound - x0[j]) / dx[j]);
			upper = x0[j] + (steps * dx[j]);
		}	
		
		//Calculate number of grid points per dimension	
		tot_ev_per_dim[j] = rint((upper - lower[j])/dx[j]) + 1;	
		total_ev *= tot_ev_per_dim[j] ;				
	}		
																				
	//For each gridpoint in dimensions 3 to N			
	for(j = 0; j < total_ev; j++)
	{							
		//Calculate location of grid point
		int divisor;
		int eval_point = j;
		for(u = 2; u < dim-1; u++)
		{			
			divisor = 1;
			for(w = u+1; w < dim; w++)
				divisor *= tot_ev_per_dim[w];
			
			gridpoint[u] = lower[u] + (dx[u] * (eval_point / divisor));			
			eval_point = eval_point % divisor;			
		}
		gridpoint[dim-1] = lower[dim-1] + (dx[dim-1] * eval_point); //Last case			
																						    
		//Fill structure with gridpoint position                        
		for(l = 2; l < dim; l++)
			dif_pos[l] = (gridpoint[l] - x0[l])/ dx[l];

		/* This code calculates, a 2D plane formed by the first two dimensions of the space, the optimal
		 * box inside the initial bounding box */

		Z = gridpoint[2] - sample[2];

		//X,Y, along with X2,XY,Y2 form the equation of the 2D rotated plane

		F = Sm1->me[2][2] * Z * Z - 1;

		X  =  2*Sm1->me[0][2]*Z*cosTheta + 2*Sm1->me[1][2]*Z*sinTheta;
		Y  = -2*Sm1->me[0][2]*Z*sinTheta + 2*Sm1->me[1][2]*Z*cosTheta;

		//Calculate displacements and obtain formula (x-xo)^2 / a^2 + % (y-yo)^2/b^2 = 1
		
		termX2 = (X/X2)/2;
		termY2 = (Y/Y2)/2; 
		valor = -F + termX2*termX2*X2 + termY2*termY2*Y2;

		//Calculate new rotated bounding box. UP and RIGHT are the corners of the new bounding box

		upy = sqrt(1/(Y2/valor)) ;
		rightx = sqrt(1/(X2/valor)) ;
		
		upx_rot    =  0 * cosTheta + upy * sinTheta;
		upy_rot    = -0 * sinTheta + upy * cosTheta;
		rightx_rot =  rightx * cosTheta + 0 * sinTheta;
		righty_rot = -rightx * sinTheta + 0 * cosTheta;
				
		//Calculate original displacement (rotated ellipse)	
				
		box_center[0] = termX2*cosTheta-termY2*sinTheta;
		box_center[1] = termX2*sinTheta+termY2*cosTheta;
		
		bound[0] = sqrt(upx_rot*upx_rot+rightx_rot*rightx_rot);
		bound[1] = sqrt(upy_rot*upy_rot+righty_rot*righty_rot);
			
		//Calculate lower and upper bound of new BoundingBox	
		for(u = 0; u < 2; u++)
		{
			box_min[u] = (sample[u] - box_center[u]) - bound[u]; 
			box_steps[u] = floor((box_min[u] - x0[u]) / dx[u]);
			lower[u] = (x0[u] > box_min[u])?(x0[u]):(x0[u] + (box_steps[u] * dx[u]));

			box_max[u] = (sample[u] - box_center[u]) + bound[u]; 
			box_steps[u] = ceil((box_max[u] - x0[u]) / dx[u]); 
			box_upper[u] = (x1[u] < box_max[u])?(x1[u]):(x0[u] + (box_steps[u] * dx[u])); 

			tot_ev_per_dim[u] = rint((box_upper[u] - lower[u])/dx[u]);			
		}
	
	    //Calculate the PDF of the defined 2D box
	    //compute2DBox_3D(pdf,PC,l(wer,tot_ev_per_dim,gridpoint,dif_pos,x0,dx,h2,cd,eigenvectors,densValues,densPosition);	
	    compute2DBox_ND(pdf,PC,lower,tot_ev_per_dim,gridpoint,dif_pos,x0,dx,dim,h2,cd,eigenvectors,densValues,densPosition,PCdot_vec,temp_vec,gridpoint_vec);	
	
	}//End of "per gridpoint" for

  } //End of "per sample" for
    
  //Delete memory structures created by threads   
  free(lower);	
  free(tot_ev_per_dim);				
  free(dif_pos);  
  free(gridpoint);       
  

  #ifdef _OPENMP
  free(densValues);
  free(densPosition);
  free(PCdot_vec);
  free(temp_vec);
  free(gridpoint_vec);   
  #else
  free(PCdot_vec);
  #endif  

  //Apply k constant to PDF
  #pragma omp for
  for(i=0; i < pdf->total_size; i++)
  	  pdf->PDF[i] = pdf->PDF[i] * k;

  //Calculate integral of PDF  
  #pragma omp for reduction(+:theintegral) 
  for(i=0; i < pdf->total_size; i++)
      theintegral += pdf->PDF[i];

  #pragma omp single
  theintegral = theintegral * total_dx;
  
  //Renormalize PDF using integral
  #pragma omp for
  for(i=0; i < pdf->total_size; i++)
     pdf->PDF[i] = pdf->PDF[i]/theintegral;
  
  //Calculate total volume of renormalized PDF     
  #pragma omp for reduction(+:total_vol)  
  for(i=0; i < pdf->total_size; i++)
	total_vol += pdf->PDF[i];

  }//End of parallel OpenMP Region   
  
  printf("Total integrated PDF: %g. The integral: %f\n",total_vol*total_dx,theintegral);

  return 0;
}

