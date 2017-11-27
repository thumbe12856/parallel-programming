#include<stdio.h>
#include<stdlib.h>
int Pow(long long int x, long long int n, long long int mod) {
	long long int Ans = 1, t = x;
	while(n) {
		if(n&1)
			Ans *= t, Ans %= mod;
		t *= t, t %= mod, n >>= 1;
	}
	return (long long int)Ans;
}

int JudgePrime(long long int n) {
	if(n == 2 || n == 3)    return 1;
	if(n == 1)    return 0;
	if(!(n&1))    return 0;

	long long int a, x, flag = 1, t;
	x = 100%(n-4)+2;
	t = Pow(x, n-1, n);

	if(t != 1)    return 0;

	return 1;
}

int main ( int argc , char * argv [])
{
	long long int pc , /* prime counter */
	     foundone ; /* most recent prime found */
	long long int n , limit ;
	sscanf(argv[1], "%llu", &limit);

	printf("Starting. Numbers to be scanned = %lld \n", limit);

	if(limit >= 2) {
		pc = 1;
		foundone = 2;
	} else {
		pc = 0;
		foundone = 0;
	}

	for(n=3; n<=limit; n++) {
		if(JudgePrime(n)) {
			pc++;
			foundone = n;
		}
	}

	printf("Done. Largest prime is %d Total primes %d\n", foundone, pc);
	return 0;
}

