#include<stdio.h>
#include<stdlib.h>
int Pow(long long x, int n, long long mod) {
    long long Ans = 1, t = x;
    while(n) {
        if(n&1)
            Ans *= t, Ans %= mod;
        t *= t, t %= mod, n >>= 1;
    }
    return (int)Ans;
}

int JudgePrime(int n) {
    if(n == 2 || n == 3)    return 1;
    if(n == 1)    return 0;
    if(!(n&1))    return 0;
    
    int a, x, flag = 1, t;
    x = rand()%(n-4)+2;
    t = Pow(x, n-1, n);
    if(t != 1)    return 0;
    
    return 1;
}

int main() {
    long long int pc , /* prime counter */
	    foundone ; /* most recent prime found */
	long long int n , limit ;
	sscanf(argv[1], "%llu", &limit);
    printf("Starting. Numbers to be scanned = %lld \n", limit);

	for (n =11; n <= limit ; n = n +2) {
		if ( JudgePrime ( n ) ) {
			pc ++;
			foundone = n ;
		}
	}
	printf("Done. Largest prime is %d Total primes %d\n", foundone, pc);

    return 0;
}