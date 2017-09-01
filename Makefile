#######################

# Makefile used to build the set of programs used to derive climate 
# models' performance indices using multidimensional kernel-based 
# probability density functions, as described by:

# Multi-objective climate model evaluation by means of multivariate kernel 
# density estimators: efficient and multi-core implementations, by
# Unai Lopez-Novoa, Jon Saenz, Alexander Mendiburu, Jose Miguel-Alonso, 
# Inigo Errasti, Agustin Ezcurra, Gabriel Ibarra-Berastegi, 2014.

# Copyright (c) 2014, Unai Lopez-Novoa, Jon Saenz, Alexander Mendiburu 
# and Jose Miguel-Alonso  (from Universidad del Pais Vasco/Euskal 
# 		    Herriko Unibertsitatea)

# Refer to README.txt for more information

#######################

#### C COMPILER ####
CC=gcc

#### LIBRARY PATHS ####

#Output path for program binaries
BINDIR=bin

#Meschach library
MESCH_INC=/home/unai/libs/mesch12b
MESCH_LIB=/home/unai/libs/mesch12b/meschach.a

#NetCDF library
NETCDF_INC=/usr/local/include
NETCDF_LIB=/usr/local/lib

#OpenCL library
OCL_LIBS = -L/opt/AMDAPPSDK-2.9-1/lib/x86_64 -lOpenCL
OCL_INC  = -I/opt/AMDAPPSDK-2.9-1/include

#######################

#### COMPILER FLAGS ####

#DEBUG=-O2 -ftree-vectorize -msse2 -fopenmp
DEBUG=-O2 -fopenmp

####################

MOBJS = $(BINDIR)/mpdfestimator.o $(BINDIR)/MPDFEstimator.o \
	$(BINDIR)/mpdfncopers.o $(BINDIR)/parseargs.o $(BINDIR)/linalg.o \
	$(BINDIR)/boundaries.o $(BINDIR)/copycenter.o $(BINDIR)/PDF.o \
	$(BINDIR)/computePDF.o $(BINDIR)/opencl_util.o 

MBSOBJS = $(BINDIR)/mpdfestimator_bootstrap.o $(BINDIR)/MPDFEstimator.o \
	$(BINDIR)/parseargs.o $(BINDIR)/linalg.o $(BINDIR)/bootstrap.o \
	$(BINDIR)/boundaries.o $(BINDIR)/copycenter.o $(BINDIR)/PDF.o \
	$(BINDIR)/computePDF.o 	$(BINDIR)/genepsilon.o $(BINDIR)/opencl_util.o

all : $(BINDIR)/mpdfestimator  $(BINDIR)/mpdfestimator-bootstrap \
	$(BINDIR)/mpdf_score


XRANGE=-20/-15/-15:25/30/30:0.2/0.2/0.2
check :
	${BINDIR}/mpdfestimator -p 0 -h 0.6 -x ${XRANGE} sample-data/sample-data

bscheck :
	${BINDIR}/mpdfestimator-bootstrap -h 0.6 -H 0.1/1/0.1\
		-x ${XRANGE} -n 5 sample-data/sample-data


CCFLAGS= ${DEBUG} -Wall -I${MESCH_INC} -I${NETCDF_INC} ${OCL_INC}

$(BINDIR)/mpdf_score: score.c
	$(CC) -Wall ${DEBUG} -o $@ score.c -lm -L/usr/lib -lnetcdf -I${NETCDF_INC}

$(BINDIR)/mpdfestimator-bootstrap : $(MBSOBJS)
	$(CC) -Wall ${DEBUG} -o $@ $(MBSOBJS) -lm -L/usr/lib ${MESCH_LIB} $(OCL_LIBS)

$(BINDIR)/mpdfestimator : $(MOBJS) 
	$(CC) -Wall ${DEBUG} -o $@ $(MOBJS) -lm -L/usr/lib ${MESCH_LIB} -lnetcdf  $(OCL_LIBS)

$(BINDIR)/mpdfestimator.o : mpdfestimator_main.c mpdfncopers.h parseargs.h linalg.h copycenter.h PDF.h computePDF.h MPDFEstimator.h opencl_util.h
	$(CC) $(CCFLAGS) -c -o $@ mpdfestimator_main.c

$(BINDIR)/mpdfestimator_bootstrap.o : mpdfestimator_bootstrap.c parseargs.h linalg.h copycenter.h PDF.h computePDF.h genepsilon.h MPDFEstimator.h boundaries.h PDF.h computePDF.h bootstrap.h opencl_util.h
	$(CC) $(CCFLAGS) -c -o $@ mpdfestimator_bootstrap.c

$(BINDIR)/PDF.o : PDF.c PDF.h 
	$(CC) $(CCFLAGS) -c -o $@ PDF.c
$(BINDIR)/genepsilon.o : genepsilon.c genepsilon.h MPDFEstimator.h 
	$(CC) $(CCFLAGS) -c -o $@ genepsilon.c
$(BINDIR)/bootstrap.o : bootstrap.c bootstrap.h MPDFEstimator.h genepsilon.h\
			PDF.h
	$(CC) $(CCFLAGS) -c -o $@ bootstrap.c  -I${MESCH_INC}

$(BINDIR)/parseargs.o : parseargs.c parseargs.h 
	$(CC) $(CCFLAGS) -c -o $@ parseargs.c
$(BINDIR)/linalg.o : linalg.c linalg.h MPDFEstimator.h
	$(CC) $(CCFLAGS) -c -o $@ -I${MESCH_INC} linalg.c
$(BINDIR)/copycenter.o : copycenter.c copycenter.h MPDFEstimator.h
	$(CC) $(CCFLAGS) -c -o $@ -I${MESCH_INC} copycenter.c
$(BINDIR)/boundaries.o : boundaries.c boundaries.h 
	$(CC) $(CCFLAGS) -c -o $@ -I${MESCH_INC} boundaries.c

$(BINDIR)/MPDFEstimator.o : MPDFEstimator.c MPDFEstimator.h linalg.h
	$(CC) $(CCFLAGS) -c -o $@ MPDFEstimator.c

$(BINDIR)/computePDF.o : computePDF.c computePDF.h linalg.h MPDFEstimator.h PDF.h
	$(CC) $(CCFLAGS) -c -o $@ computePDF.c

$(BINDIR)/mpdfncopers.o : mpdfncopers.c mpdfncopers.h  
	$(CC) $(CCFLAGS) -c -o $@ mpdfncopers.c

$(BINDIR)/opencl_util.o : opencl_util.c opencl_util.h
	$(CC) $(CCFLAGS) $(OCL_INC) -c -o $@ opencl_util.c

clean :
	\rm -f $(BINDIR)/mpdfestimator $(BINDIR)/mpdfestimator-bootstrap $(BINDIR)/mpdf_score ${MOBJS} ${MBSOBJS}


