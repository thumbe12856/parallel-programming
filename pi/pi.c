#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

struct info {
	long long int times;
	long long int circleSum;
};

//long long int circleSum = 0;
//long long int times;

static unsigned long x=123456789, y=362436069, z=521288629;

unsigned long xorshf96(void) {          //period 2^96-1
	unsigned long t;
	x ^= x << 16;
	x ^= x >> 5;
	x ^= x << 1;

	t = x;
	x = y;
	y = z;
	z = t ^ x ^ y;

	return z;
}

void *monteCarlo(void *ptr)
{
	struct info* p = (struct info*) ptr;
	long long int i, times = p->times, circleSum = 0;
	double x, y;
	srand(time(NULL));
	unsigned int seed = rand();

	for(i=0; i<times; i++) {
		// 0.0 ~ 0.99999 -> 0.0 ~ 1.99999 -> -1.0 ~ 0.99999
		x = (rand_r(&seed) / (double)(RAND_MAX)) * 2 - 1;
		y = (rand_r(&seed) / (double)(RAND_MAX)) * 2 - 1;
		if(x*x + y*y <= 1) {
			circleSum++;
		}
	}
	p->circleSum = circleSum;
	pthread_exit(NULL);	    
	return NULL;
}

int main(int argc, char *argv[])
{
	if(argc != 3) {
		printf("input error!\n");
		return 0;
	}

	long long int cpuCore, executeTimes;
	cpuCore = atoi(argv[1]);
	executeTimes = atoi(argv[2]);

	pthread_t thread[cpuCore];
	struct info p[cpuCore];
	int times = executeTimes / cpuCore;

	long long int i;
	for(i=0; i<cpuCore; i++) {
		p[i].times = times;
		//printf("Create: thread:%d, circle sum: %d\n", i, p[i].circleSum);
		pthread_create(&thread[i], NULL, monteCarlo, &p[i]);
		//pthread_create(&thread[i], NULL, monteCarlo, NULL);
	}

	long long int finalSum = 0, finalTimes = 0;
	for(i=0; i<cpuCore; i++) {
		pthread_join(thread[i], NULL);
		finalTimes += times;
		finalSum += p[i].circleSum;
		//printf("Join: thread:%d, circle sum: %d\n", i, p[i].circleSum);
	}

	printf("pi_estimate:%lf\n\n", 4 * finalSum / (double)finalTimes);

	return 0;
}
