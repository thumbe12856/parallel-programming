#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

long long int sqrtLongInt = 3037000500;
long long int *mark;
long long int *prime;

void erase(){
	long long int i, j;
	long long int sq = 55109;
		fprintf(stderr, "here");
	mark[1] = 1;
		fprintf(stderr, "here");
	for (i=2; i<=sq; i++){

		if (!mark[i]){
			for (j=i*i;j<=sqrtLongInt;j+=i){
				mark[j] = 1;
			}
		}
	}
}

int main(int argc, char *argv[]) {
	long long int n,j,flag,sq,i=0;
	mark = malloc(sqrtLongInt * sizeof(int));
	prime = malloc(sqrtLongInt * sizeof(int));
	//memset(mark, 0, sqrtLongInt);
		fprintf(stderr, "here");
	erase();

	for(n=2;n<sqrtLongInt;n++){
		if (!mark[n]){
			prime[i]=n;
			i++;                
		}  
	}

	prime[i]=9223372036854775807;

	long long int limit;
	sscanf(argv[1], "%llu", &limit);
	printf("Starting. Numbers to be scanned = %lld \n", limit);

	long long int pc, foundone;
	if(limit>=3) {
		pc = 1;
	} else if(limit==2) {
		pc = 1;
		foundone = 2;
	} else {
		pc = 0;
		foundone = 0;
	}

	for(n=3; n<=limit; n++) {
		flag=0;
		if (((!(n%2))&&(n>2))||(n==1)) flag=1;
		else {
			j=1;
			sq=(int)sqrt(n);
			while (prime[j]<= sq){
				if ((n%prime[j])==0) {
					flag=1;
					break;  
				}
				j++;   
			}
		}

		if (flag != 1) {
			pc++;
			foundone = n;
		}
	}
	
	printf("Done. Largest prime is %d Total primes %d\n", foundone, pc);
	return 0;
}

