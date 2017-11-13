#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
static long num_steps = 100000000;
double step;
#define NUM_THREADS 4

int main () {
	/*
	int i, nthreads;
	double x, pi, sum[NUM_THREADS];
	step = 1.0/(double) num_steps;
	omp_set_num_threads(NUM_THREADS);

	#pragma omp parallel
	{
		int i, id, nthrds;
		double x;
		id = omp_get_thread_num();
		nthrds = omp_get_num_threads();
		printf("%d %d\n", id, nthrds);
		if (id == 0) nthreads = nthrds;
		for (i=id, sum[id]=0.0; i< num_steps; i=i+nthrds) {
			x = (i+0.5)*step;
			sum[id] += 4.0/(1.0+x*x);
		}
	}

	for(i=0, pi=0.0; i < nthreads; i++)
		pi += sum[i] * step;

	printf("%.10lf\n", pi);
	*/
	int i,j,k=0, z[3];
	z[0] = 1;
	z[1] = 5;
	z[2] = 10;
	#pragma omp parallel for reduction (+:k)
	for(i=1; i<=100; i++)
	{
		k+=i;
	}
	printf("%d\n", k);

	return 0;
}









