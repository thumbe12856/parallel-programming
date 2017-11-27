#include<stdio.h>
#include<math.h>
#include"mpi.h"
#define PI 3.1415926535


int main ( int argc , char ** argv )
{
	long long int i, num_intervals ;
	double rect_width , area , sum , x_middle ;
	
	sscanf(argv[1], "%llu", &num_intervals);
	
	rect_width = PI / num_intervals ;
	sum = 0;
	
	int rank, size;
	int tag = 0, source, dest, p;
	MPI_Status status;
	MPI_Init( &argc, &argv);
	MPI_Comm_rank( MPI_COMM_WORLD, &rank);
	MPI_Comm_size( MPI_COMM_WORLD, &size);
	
	long long int from, to;
	to = (rank+1) * num_intervals / size;
	
	if(rank != 0) {
		from = rank * num_intervals / size;
		
		for ( i = from; i < to; i ++) {
			/* find the middle of the interval on the X - axis . */
			x_middle = ( i - 0.5) * rect_width ;
			area = sin ( x_middle ) * rect_width ;
			sum = sum + area ;
		}
		dest = 0;
		MPI_Send(&sum, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);

	} else {
		for ( i = 1; i < to; i ++) {
			/* find the middle of the interval on the X - axis . */
			x_middle = ( i - 0.5) * rect_width ;
			area = sin ( x_middle ) * rect_width ;
			sum = sum + area ;
		}
		double tempSum;
		for (source= 1; source < size; source++ ) {
			MPI_Recv(&tempSum, 1, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
			sum += tempSum;
		}
		
		printf("The total area is :%f\n", (float)sum);
	}
	
	MPI_Finalize();

	return 0;
}

