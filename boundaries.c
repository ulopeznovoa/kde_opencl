/**************

boundaries.c

Calculate boundaries of areas affected by the multidimensional Epanechnikov 
kernel.

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

#include <math.h>
#include <stdio.h>

/* Use library meschach */
#include "matrix.h"
#include "matrix2.h"

#include "boundaries.h"

//Output: out vector, with max value per dimension
void calculateBoundaries(double *out, double h, VEC * eigenvals, 
			 MAT * eigenvectors,MAT *sqrtevals, MAT * temp_bounds)
{
	int i,j; //Loop variable
	
	m_mlt(sqrtevals,eigenvectors,temp_bounds); //Multiply sqrt(Eigenvals) * Eigenvector			

	//Calculate boundaries. Per dim = bound = h * sqrt(sqr(bound))
	for(i = 0; i < eigenvals->dim; i++)
	{
		out[i] = 0;
		
		for(j=0; j < eigenvals->dim; j++)
			out[i] += pow(temp_bounds->me[j][i],2); //Read the transposed matrix
	
		out[i] = sqrt(out[i]);
		out[i] *= h;
	}	
}
