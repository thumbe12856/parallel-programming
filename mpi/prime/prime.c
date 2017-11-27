#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mpi.h"

int isprime ( long long int n ) {
	long long int i , squareroot ;

	if(n == 3 || n == 5 || n==7) return 1;
	if (n >10) {
		squareroot = ( long long int ) sqrt ( n ) ;
		for ( i =3; i <= squareroot ; i = i +2) {
			if (( n % i ) ==0)
				return 0;
		}
		return 1;
	}
	else
		return 0;
}
int main ( int argc , char * argv [])
{
	int i,j,k;
	int rank, size;
	MPI_Init( &argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank( MPI_COMM_WORLD, &rank);

	long long int pc , /* prime counter */
	    foundone ; /* most recent prime found */
	long long int n , limit ;
	sscanf(argv[1], "%llu", &limit);
	if(rank == 0)
		printf("Starting. Numbers to be scanned = %lld \n", limit);

	pc = 4; /* Assume (2 ,3 ,5 ,7) are counted here */
	
	
	long long int from, to;
	long long int pcc[2] =  {0};
	MPI_Status status;
	int tag=0, source, dest, p;

	to = (rank + 1) * limit / size;
	if(rank != 0) {
		from = rank * limit / size + 1;
		if(!(from&1)) from = from + 1;
		for (n = from; n <= to; n = n +2) {
			if ( isprime ( n ) ) {
				pcc[0] ++;
				pcc[1] = n;
			}
		}
		//printf("Done %d. Largest prime is %d Total primes %d\n", rank, pcc[1], pcc[0]);
		dest = 0;
		MPI_Send(pcc, 2, MPI_LONG_LONG_INT, dest, tag, MPI_COMM_WORLD);
	
	} else {
		if(limit >= 2) {
			pcc[0] += 1;
			pcc[1] = 2;
		}
		for (n = 3; n <= to; n = n +2) {
			if ( isprime ( n ) ) {
				pcc[0] ++;
				pcc[1] = n;
			}
		}

		long long int temp[2];
		for(source=1; source<size; source++) {
			MPI_Recv(&temp, 2, MPI_LONG_LONG_INT, source, tag, MPI_COMM_WORLD, &status);
			//printf("%d %d\n", temp[0], temp[1]);
			pcc[0] += temp[0];
			if(temp[1] > pcc[1]) pcc[1] = temp[1];
		}
		
		printf("Done. Largest prime is %d Total primes %d\n", pcc[1], pcc[0]);
	}

	MPI_Finalize();
	return 0;
}

