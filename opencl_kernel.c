
#pragma OPENCL EXTENSION cl_khr_fp64: enable

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

// KDE Kernels 2D

__kernel void calc_gp_2D(
const int nsamples,
__global double * samples,
__global double * x0,
__global double * x1,
__global double * dx,
__global double * bounds,

__global int * gp_number_x, //OUT: Number of gridpoints in x axis
__global int * gp_number_y, //OUT: Number of gridpoints in y axis
__global int * gp_number_tot, //OUT: Number of gridpoints in x and y axis
__global double * gp_lower_x, //OUT: Lower bound of gridpoints in x axis
__global double * gp_lower_y //OUT: Lower bound of gridpoints in y axis

)
{
	int tId = get_global_id(0);
	
	if(tId >= nsamples) return;

	int i,steps;
	double abs_bound,upper;

	double sample[2];
	sample[0] = samples[tId * 2 + 0];
	sample[1] = samples[tId * 2 + 1];

	double gp_lower[2];
	int gp_number[2];

	//For each sample, calculate its bounding box, 
	//expressed as coordinates of lower corner and number of gridpoints per dimensions
	for(i = 0; i < 2; i++)
	{	
		//Lower corner
		abs_bound = sample[i] - bounds[i];
		if (x0[i] > abs_bound)
			gp_lower[i] = x0[i];
		else
		{
			steps = floor((abs_bound - x0[i]) / dx[i]);
			gp_lower[i] = x0[i] + (steps * dx[i]);
		}

		//Upper corner
		abs_bound = sample[i] + bounds[i];
		if (x1[i] < abs_bound)
			upper = x1[i];
		else
		{
			steps = ceil((abs_bound - x0[i]) / dx[i]);
			upper = x0[i] + (steps * dx[i]);
		}	
		
		//Calculate number of eval points per dimension	
		gp_number[i] = rint((upper - gp_lower[i])/dx[i]) + 1;			
	}    

	//Store
	gp_lower_x[tId] = gp_lower[0];
	gp_lower_y[tId] = gp_lower[1];

	gp_number_x[tId] = gp_number[0];
	gp_number_y[tId] = gp_number[1];
	gp_number_tot[tId] = gp_number[0] * gp_number[1];
}

__kernel void density_2D(
const int offset,	//IN: Offset in the samples 
const int nsamples,	//IN: Number of samples to be processed 
const double cd,		//IN: Volume constant value
const double h2,
__constant double * eigenvectors, //IN: Eigenvectors matrix
__constant double * x0, //IN: lower bounds of evaluation space
__constant double * dx, //IN: deltas of evaluation space
__constant long * pdfcumsize, //IN: accumulated size for pdf

__global double * samples_pc, //IN: dataset samples

__global int * gp_number_x, //IN: Number of gridpoints in x axis
__global int * gp_number_y, //IN: Number of gridpoints in y axis
__global double * gp_lower_x, //IN: Lower bound of gridpoints in x axis
__global double * gp_lower_y, //IN: Lower bound of gridpoints in y axis
__global int * gp_number_xy_scan, //IN: Number of gridpoints in x and y axis

__global double * density_values, //OUT:
__global int * density_positions //OUT:
)
{	
	int tId = get_global_id(0);

	if(tId >= nsamples) return;

	int i,j;
	int sample_id = tId + offset;

	int gp_x = gp_number_x[sample_id];
	int gp_y = gp_number_y[sample_id];

	double gridpoint[2];
	double lower_b[2];

	lower_b[0] = gp_lower_x[sample_id];
	lower_b[1] = gp_lower_y[sample_id];
		
	int initial_position = gp_number_xy_scan[sample_id]; 
	
	double PC[2];
	double PCdot[2];
		
	PC[0] = samples_pc[sample_id * 2 + 0];
	PC[1] = samples_pc[sample_id * 2 + 1];

	int dif_pos[2];

	for(i = 0; i < gp_x; i++)
		for(j = 0; j < gp_y; j++)
		{			
			//Calculate gridpoint coordinates 
			gridpoint[0] = lower_b[0] + (dx[0] * i);
			gridpoint[1] = lower_b[1] + (dx[1] * j);

			//Conversion to PC space			
			PCdot[0] = (eigenvectors[0] * gridpoint[0]) + (eigenvectors[1] * gridpoint[1]);
			PCdot[1] = (eigenvectors[2] * gridpoint[0]) + (eigenvectors[3] * gridpoint[1]);	

			//Absolute distance calculation
			double temp = (((PC[0] - PCdot[0]) * (PC[0] - PCdot[0])) + ((PC[1] - PCdot[1]) * (PC[1] - PCdot[1]))) / h2;

			//PDFposition			
			int position = initial_position + (gp_y * i) + j; //Position in the OpenCL device vector
																	
			//Position to store the density value in host memory								
			dif_pos[0] = convert_int_rte((gridpoint[0] - x0[0])/ dx[0]);
			dif_pos[1] = convert_int_rte((gridpoint[1] - x0[1])/ dx[1]); 
						
			density_positions[position] = (dif_pos[0] * pdfcumsize[0]) + (dif_pos[1] * pdfcumsize[1]);
							
			//Density value
			density_values[position] = (0.5/cd*(2+2.)*(1.-temp)) * (fabs(temp)<1.);	 	
		}	
}

// KDE Kernels 3D

__kernel void calc_slices(
const int nsamples,
__global double * samples,
__global int * slices, 
__global double * slices_lower,

__global double * x0,
__global double * x1,
__global double * dx,
__global double * bounds
)
{
	int tId = get_global_id(0);
	
	if(tId >= nsamples) return;

	double sample = samples[tId * 3 + 2];
	double abs_bound, lower, upper;
	int steps;

	//Calculate boundaries for Z axis
	//Lower corner
	abs_bound = sample - bounds[2];
	if (x0[2] > abs_bound)
		lower = x0[2];
	else
	{
		steps = ceil((abs_bound - x0[2]) / dx[2]);
		
		lower = x0[2] + (steps * dx[2]);
	}

	//Upper corner
	abs_bound = sample + bounds[2];
	if (x1[2] < abs_bound)
		upper = x1[2];
	else
	{
		steps = floor((abs_bound - x0[2]) / dx[2]);
	
		upper = x0[2] + (steps * dx[2]);
	}	

	
	//Calculate number of grid points per dimension	
	slices[tId] = rint((upper - lower)/dx[2]) + 1;	
	slices_lower[tId] = lower;
}

__kernel void calc_gp(
const int sample_offset,	 //IN: Id of sample to be processed
__constant croppingVars * cp, //IN: Struct with constants for geometric croppping
__constant double * sm1, //IN: Inverse of covariance matrix
__constant double * x0, //IN: lower bounds of evaluation space
__constant double * x1, //IN: upper bounds of evaluation space
__constant double * dx, //IN: deltas of evaluation space

__global double * samples, //IN: dataset samples
__global int * slices, //IN: number of slices per sample
__global int * slices_scan, //IN: Exc-scan of previous vector
__global double * slices_lower, //IN: lower bounds of z axis slices per sample

__global int * gp_number_x, //OUT: Number of gridpoints in x axis
__global int * gp_number_y, //OUT: Number of gridpoints in y axis
__global int * gp_number_tot, //OUT: Number of gridpoints in x and y axis
__global double * gp_lower_x, //OUT: Lower bound of gridpoints in x axis
__global double * gp_lower_y, //OUT: Lower bound of gridpoints in y axis

__global int * sample_index,
__global int * slice_index

)
{
	int sample_id = get_global_id(0) + sample_offset;
	int slice_id = get_global_id(1);

	if(slice_id >= slices[sample_id]) return;

	//Calculate location of grid point
	double slice  = slices_lower[sample_id] + (dx[2] * slice_id); 			

	double sample[3];
	sample[0] = samples[sample_id * 3 + 0];
	sample[1] = samples[sample_id * 3 + 1];
	sample[2] = samples[sample_id * 3 + 2];

	double Z = slice - sample[2];

	//X,Y, along with X2,XY,Y2 form the equation of the 2D rotated plane

	double F = sm1[8] * Z * Z - 1;

	double X  =  2 * sm1[2] * Z * cp->cosTheta + 2 * sm1[5] * Z * cp->sinTheta; 
	double Y  = -2 * sm1[2] * Z * cp->sinTheta + 2 * sm1[5] * Z * cp->cosTheta; 

	//Calculate displacements and obtain formula (x-xo)^2 / a^2 + % (y-yo)^2/b^2 = 1
	
	double termX2 = (X/cp->X2)/2;
	double termY2 = (Y/cp->Y2)/2; 
	double valor = -F + termX2 * termX2 * cp->X2 + termY2 * termY2 * cp->Y2;

	//Calculate new rotated bounding box. UP and RIGHT are the corners of the new bounding box
	
	double upy = sqrt(1/(cp->Y2/valor));
	double rightx = sqrt(1/(cp->X2/valor));
	
	double upx_rot    =  0 * cp->cosTheta + upy * cp->sinTheta;
	double upy_rot    = -0 * cp->sinTheta + upy * cp->cosTheta;
	double rightx_rot =  rightx * cp->cosTheta + 0 * cp->sinTheta;
	double righty_rot = -rightx * cp->sinTheta + 0 * cp->cosTheta;
	
	//Calculate original displacement (rotated ellipse)	
	double box_center[2]; 					
	box_center[0] = termX2 * cp->cosTheta - termY2 * cp->sinTheta;
	box_center[1] = termX2 * cp->sinTheta + termY2 * cp->cosTheta;
	
	double bound[2];
	bound[0] = sqrt(upx_rot*upx_rot+rightx_rot*rightx_rot);
	bound[1] = sqrt(upy_rot*upy_rot+righty_rot*righty_rot);
		
	double box_upper[2];	
	double box_max[2];	
	double box_min[2];
	double box_steps[2];	
		
	//Calculate lower and upper bound of new BoundingBox	
	
	double lower_x_reg, lower_y_reg;
	int tot_x, tot_y;
	
	//Dim x

	box_min[0] = (sample[0] - box_center[0]) - bound[0]; 
	box_steps[0] = floor((box_min[0] - x0[0]) / dx[0]);
	lower_x_reg = (x0[0] > box_min[0])?(x0[0]):(x0[0] + (box_steps[0] * dx[0]));

	box_max[0] = (sample[0] - box_center[0]) + bound[0]; 
	box_steps[0] = ceil((box_max[0] - x0[0]) / dx[0]); 
	box_upper[0] = (x1[0] < box_max[0])?(x1[0]):(x0[0] + (box_steps[0] * dx[0])); 

	tot_x = (round((box_upper[0] - lower_x_reg)/dx[0]));			

	//Dim y

	box_min[1] = (sample[1] - box_center[1]) - bound[1]; 
	box_steps[1] = floor((box_min[1] - x0[1]) / dx[1]);
	lower_y_reg = (x0[1] > box_min[1])?(x0[1]):(x0[1] + (box_steps[1] * dx[1]));

	box_max[1] = (sample[1] - box_center[1]) + bound[1]; 
	box_steps[1] = ceil((box_max[1] - x0[1]) / dx[1]); 
	box_upper[1] = (x1[1] < box_max[1])?(x1[1]):(x0[1] + (box_steps[1] * dx[1])); 

	tot_y = (round((box_upper[1] - lower_y_reg)/dx[1]));	

	//Store
	int first = slices_scan[sample_offset];
	int pos = slices_scan[sample_id] - first + slice_id; 

	gp_number_x[pos] = tot_x;
	gp_number_y[pos] = tot_y;
	gp_number_tot[pos] = tot_x * tot_y;	

	gp_lower_x[pos] = lower_x_reg;
	gp_lower_y[pos] = lower_y_reg;

	sample_index[pos] = sample_id;
	slice_index[pos] = slice_id;
}

__kernel void density(
const int total_slices,	//IN: Number of slices to be processed in this chunk

const double cd,		//IN: Volume constant value
const double h2,
__constant double * eigenvectors, //IN: Eigenvectors matrix
__constant double * x0, //IN: lower bounds of evaluation space
__constant double * dx, //IN: deltas of evaluation space
__constant long * pdfcumsize, //IN: accumulated size for pdf

__global double * samples_pc, //IN: dataset samples
__global double * slices_lower, //IN: lower bounds of z axis slices per sample

__global int * gp_number_x, //IN: Number of gridpoints in x axis
__global int * gp_number_y, //IN: Number of gridpoints in y axis
__global double * gp_lower_x, //IN: Lower bound of gridpoints in x axis
__global double * gp_lower_y, //IN: Lower bound of gridpoints in y axis
__global int * gp_number_xy_scan, //IN: Number of gridpoints in x and y axis

__global int * sample_index, //IN:
__global int * slice_index, //IN:

__global double * density_values, //OUT:
__global int * density_positions //OUT:
)
{	
	int tId = get_global_id(0);

	if(tId >= total_slices) return;

	int sample_id = sample_index[tId];
	int slice_id = slice_index[tId];

	int i,j;

	int gp_x = gp_number_x[tId];
	int gp_y = gp_number_y[tId];

	double gridpoint[3];
	double lower_b[3];

	lower_b[0] = gp_lower_x[tId];
	lower_b[1] = gp_lower_y[tId];
	lower_b[2] = slices_lower[sample_id];
		
	int initial_position = gp_number_xy_scan[tId]; 
	
	double PC[3];
	double PCdot[3];
		
	PC[0] = samples_pc[sample_id * 3 + 0];
	PC[1] = samples_pc[sample_id * 3 + 1];
	PC[2] = samples_pc[sample_id * 3 + 2];

	int dif_pos[3];

	for(i = 0; i < gp_x; i++)
		for(j = 0; j < gp_y; j++)
		{			
			//Calculate gridpoint coordinates 
			gridpoint[0] = lower_b[0] + (dx[0] * i);
			gridpoint[1] = lower_b[1] + (dx[1] * j);
			gridpoint[2] = lower_b[2] + (dx[2] * slice_id); 

			//Conversion to PC space			
			PCdot[0] = (eigenvectors[0] * gridpoint[0]) + (eigenvectors[1] * gridpoint[1]) + (eigenvectors[2] * gridpoint[2]);
			PCdot[1] = (eigenvectors[3] * gridpoint[0]) + (eigenvectors[4] * gridpoint[1]) + (eigenvectors[5] * gridpoint[2]);	
			PCdot[2] = (eigenvectors[6] * gridpoint[0]) + (eigenvectors[7] * gridpoint[1]) + (eigenvectors[8] * gridpoint[2]);	

			//Absolute distance calculation
			double temp = (((PC[0] - PCdot[0]) * (PC[0] - PCdot[0])) 
						 + ((PC[1] - PCdot[1]) * (PC[1] - PCdot[1])) 
						 + ((PC[2] - PCdot[2]) * (PC[2] - PCdot[2]))) / h2;

			//PDFposition			
			int position = initial_position + (gp_y * i) + j; //Position in the OpenCL device vector
																	
			//Position to store the density value in host memory								
			dif_pos[0] = convert_int_rte((gridpoint[0] - x0[0])/ dx[0]);
			dif_pos[1] = convert_int_rte((gridpoint[1] - x0[1])/ dx[1]); 
			dif_pos[2] = convert_int_rte((gridpoint[2] - x0[2])/ dx[2]);	
						
			density_positions[position] = (dif_pos[0] * pdfcumsize[0]) + (dif_pos[1] * pdfcumsize[1]) + (dif_pos[2] * pdfcumsize[2]);
							
			//Density value
			density_values[position] = (0.5/cd*(3+2.)*(1.-temp)) * (fabs(temp)<1.);	 			

			//DEBUG
/*			int pos = (dif_pos[0] * pdfcumsize[0]) + (dif_pos[1] * pdfcumsize[1]) + (dif_pos[2] * pdfcumsize[2]);
			double val = (0.5/cd*(3+2.)*(1.-temp)) * (fabs(temp)<1.);	 			

			printf("\ntId %d sample %d slice %d gp %d,%d pos %d val %f",tId,sample_id,slice_id,i,j,pos,val);
*/	
		}	
}

// SCAN Kernels (from SHOC)

__kernel void
reduce(__global const int * in,
       __global int * isums,
       const int n,
       __local int * lmem)
{
    // First, calculate the bounds of the region of the array
    // that this block will sum.  We need these regions to match
    // perfectly with those in the bottom-level scan, so we index
    // as if vector types of length 4 were in use.  This prevents
    // errors due to slightly misaligned regions.
    int region_size = ((n / 4) / get_num_groups(0)) * 4;
    int block_start = get_group_id(0) * region_size;

    // Give the last block any extra elements
    int block_stop  = (get_group_id(0) == get_num_groups(0) - 1) ?
        n : block_start + region_size;

    // Calculate starting index for this thread/work item
    int tid = get_local_id(0);
    int i = block_start + tid;

    int sum = 0.0f;

    // Reduce multiple elements per thread
    while (i < block_stop)
    {
        sum += in[i];
        i += get_local_size(0);
    }
    // Load this thread's sum into local/shared memory
    lmem[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduce the contents of shared/local memory
    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            lmem[tid] += lmem[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    // Write result for this block to global memory
    if (tid == 0)
    {
        isums[get_group_id(0)] = lmem[0];
    }
}

// This kernel scans the contents of local memory using a work
// inefficient, but highly parallel Kogge-Stone style scan.
// Set exclusive to 1 for an exclusive scan or 0 for an inclusive scan
inline int scanLocalMem(int val, __local int* lmem, int exclusive)
{
    // Set first half of local memory to zero to make room for scanning
    int idx = get_local_id(0);
    lmem[idx] = 0.0f;

    // Set second half to block sums from global memory, but don't go out
    // of bounds
    idx += get_local_size(0);
    lmem[idx] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Now, perform Kogge-Stone scan
    int t;
    for (int i = 1; i < get_local_size(0); i *= 2)
    {
        t = lmem[idx -  i]; barrier(CLK_LOCAL_MEM_FENCE);
        lmem[idx] += t;     barrier(CLK_LOCAL_MEM_FENCE);
    }
    return lmem[idx-exclusive];
}

__kernel void
top_scan(__global int * isums, const int n, __local int * lmem)
{
    int val = get_local_id(0) < n ? isums[get_local_id(0)] : 0.0f;
    val = scanLocalMem(val, lmem, 1);

    if (get_local_id(0) < n)
    {
        isums[get_local_id(0)] = val;
    }
}

__kernel void
bottom_scan(__global const int * in,
            __global const int * isums,
            __global int * out,
            const int n,
            __local int * lmem)
{
    __local int s_seed;
    s_seed = 0;

    //Exclusive Scan - First element zero
    if(get_global_id(0) == 0)
	out[0] = 0;

    // Prepare for reading 4-element vectors
    // Assume n is divisible by 4
    __global int4 *in4  = (__global int4*) in;
    //__global int4 *out4 = (__global int4*) out; //Original
    __global int4 *out4 = (__global int4*) &out[1]; //Shift one to make exclusive
    int n4 = n / 4; //vector type is 4 wide

    int region_size = n4 / get_num_groups(0);
    int block_start = get_group_id(0) * region_size;
    // Give the last block any extra elements
    int block_stop  = (get_group_id(0) == get_num_groups(0) - 1) ?
        n4 : block_start + region_size;

    // Calculate starting index for this thread/work item
    int i = block_start + get_local_id(0);
    unsigned int window = block_start;

    // Seed the bottom scan with the results from the top scan (i.e. load the per
    // block sums from the previous kernel)
    int seed = isums[get_group_id(0)];

    // Scan multiple elements per thread
    while (window < block_stop) {
        int4 val_4;
        if (i < block_stop) {
            val_4 = in4[i];
        } else {
            val_4.x = 0.0f;
            val_4.y = 0.0f;
            val_4.z = 0.0f;
            val_4.w = 0.0f;
        }

        // Serial scan in registers
        val_4.y += val_4.x;
        val_4.z += val_4.y;
        val_4.w += val_4.z;

        // ExScan sums in local memory
        int res = scanLocalMem(val_4.w, lmem, 1);

        // Update and write out to global memory
        val_4.x += res + seed;
        val_4.y += res + seed;
        val_4.z += res + seed;
        val_4.w += res + seed;

        if (i < block_stop)
        {
            out4[i] = val_4;
        }

        // Next seed will be the last value
        // Last thread puts seed into smem.
        barrier(CLK_LOCAL_MEM_FENCE);
        if (get_local_id(0) == get_local_size(0)-1) {
              s_seed = val_4.w;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Broadcast seed to other threads
        seed = s_seed;

        // Advance window
        window += get_local_size(0);
        i += get_local_size(0);
    }
}

