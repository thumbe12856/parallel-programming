#include <stdio.h>
#include <stdlib.h>

__global__ void testKernel(int param){
    printf("%d, %d\n", threadIdx.x, param);
}

int main(void){

    // initialize cuPrintf

    int N = 3;
    int a = 456;
    dim3 threadsPerBlock(N, N);

    printf("init\n");
    testKernel<<<1,threadsPerBlock>>>(a);

    return 0;
}
