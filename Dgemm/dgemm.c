#include <stdio.h>      /* standard I/O routines                 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>      /* standard I/O routines                 */
#include <omp.h>		/* Shared memory parallelism libary      */
#include <mkl.h>		/* BLAS library							 */

#include <sys/time.h>
#include "utils.h"


#ifdef PAPI
#include "cscs_papi.h"
#endif

#define NN 8192;
#define NTIMES 10


double mysecond()
{
	struct timeval tp;
	struct timezone tzp;
	int i;

	i = gettimeofday(&tp,&tzp);
	return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}




int main (int argc, char *argv[])
{

	int N = NN;
	int M = NN;
	int K = NN;
	int ii;

	/* Find out my identity in the default communicator */

	int Ncpu = 0;

	if ( argc == 2 ) 
	{
		N    = atoi(argv[1]);
		M = N;
		K = N;
	}

	double  alpha =  1.;
	double  beta  = -1.;

	int lda    = M;
	int ldb    = K;
	int ldc    = M;

	int size_A = K*lda;
	int size_B = N*ldb;
	int size_C = N*ldc;

	double *A  = (double*) malloc(sizeof(double)*size_A);
	if (A == 0) printf("Could not allocate A.\n");

	double *B  = (double*) malloc(sizeof(double)*size_B);
	if (B == 0) printf("Could not allocate B.\n");

	double *C  = (double*) malloc(sizeof(double)*size_C);
	if (C == 0) printf("Could not allocate C.\n");

	double *Cg = (double*) malloc(sizeof(double)*size_C);
	if (Cg == 0) printf("Could not allocate Cg.\n");

	fill(A,  size_A,  31.);
	eye (B,     ldb,   N );
	fill(C,  size_C,  31.);
	fill(Cg, size_C,  31.);


	int t;

	int nthreads, tid, procs, maxthreads, inpar, dynamic, nested;

	char transa = 'n';
	char transb = 'n';

	/* Get environment information */
#ifdef _OPENMP
	procs      = omp_get_num_procs();
	nthreads   = omp_get_num_threads();
	maxthreads = omp_get_max_threads();
	inpar      = omp_in_parallel();
	dynamic    = omp_get_dynamic();
	nested     = omp_get_max_active_levels();
	/* Print environment information */
	printf("Number of processors          = %d\n", procs);
	printf("Number of threads             = %d\n", nthreads);
	printf("Max threads                   = %d\n", maxthreads);
	printf("In parallel?                  = %d\n", inpar);
	printf("Dynamic threads enabled?      = %d\n", dynamic);
	printf("Max active nested levels      = %d\n", nested);
#endif

	double otime = 0.;
	otime -= mysecond();

	for (ii = 0; ii < NTIMES; ++ii)
		cblas_dgemm(CblasRowMajor, 
            CblasNoTrans, CblasNoTrans, 
            M, N, K,
            alpha, A, lda,
            B, ldb, 
            beta, C, ldc);

	otime += mysecond();
	otime /= NTIMES;

	printf("Size = %d, Gflops Max: %f  Min: %f  Avg:  %f\n", N, 2.*M*N*K/otime/1e9, 2.*M*N*K/otime/1e9, 2.*M*N*K/otime/1e9);

	free(A);
	free(B);
	free(C);
	free(Cg);

	exit(0);
}
