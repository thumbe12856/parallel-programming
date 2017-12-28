#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <CL/cl.h>
using namespace std;
#define MAX_SOURCE_SIZE (0x100000)	

int main() 
{
	//==================== INIT ====================
	
	cl_int err;
	cl_uint num;
	err = clGetPlatformIDs(0, 0, &num);
	if(err != CL_SUCCESS)
	{
		cout << "Unable to get platforms\n";
		return 0;
	}
	
	vector<cl_platform_id> platforms(num);
	err = clGetPlatformIDs(num, &platforms[0], &num);
	if(err != CL_SUCCESS)
	{
		cout << "Unable to get platform ID\n";
		return 0;
	}

	FILE *fp;
	const char fileName[] = "./hello.cl";
	size_t source_size;
	char *source_str;
 
	/* Load kernel source file */
	fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");	
		exit(1);
	}

	source_str = (char *)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	return 0;
}

