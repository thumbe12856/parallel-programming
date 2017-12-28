#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <CL/cl.h>
using namespace std;

#define MAX_SOURCE_SIZE (0x100000)	

unsigned int * histogram(unsigned int *image_data, unsigned int _size) {

	unsigned int *img = image_data;
	unsigned int *ref_histogram_results;
	unsigned int *ptr;

	ref_histogram_results = (unsigned int *)malloc(256 * 3 * sizeof(unsigned int));
	ptr = ref_histogram_results;
	memset (ref_histogram_results, 0x0, 256 * 3 * sizeof(unsigned int));


	/* OpenCL INIT	 */
	cl_int err;
	cl_uint num;
	err = clGetPlatformIDs(0, 0, &num);
	if(err != CL_SUCCESS) {
		cout << "Unable to get platforms" << endl;
		exit(1);
	}
	
	vector<cl_platform_id> platforms(num);
	err = clGetPlatformIDs(num, &platforms[0], &num);
	if(err != CL_SUCCESS) {
		cout << "Unable to get platform ID" << endl;
		exit(1);
	}
	
	/* OpenCL context */
	cl_context_properties prop[] = {CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms[0]), 0};
	cl_context context = clCreateContextFromType(prop, CL_DEVICE_TYPE_DEFAULT, NULL, NULL, NULL);
	if(context == 0) {
		cout << "Can't create OpenCL context" << endl;
		exit(1);
	}
	
	size_t cb;
	clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
	vector<cl_device_id> devices(cb / sizeof(cl_device_id));
	clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, &devices[0], 0);
	
	clGetDeviceInfo(devices[0], CL_DEVICE_EXTENSIONS, 0, NULL, &cb);
	string devname;
	devname.resize(cb);
	clGetDeviceInfo(devices[0], CL_DEVICE_EXTENSIONS, cb, &devname[0], 0);
			
	/* OpenCL command Queue	*/
	cl_command_queue queue = clCreateCommandQueueWithProperties(context, devices[0], 0, 0);
	if(queue == 0) {
		cout << "Can't create command queue" << endl;
		clReleaseContext(context);
		exit(1);
	}
	
	int total_size[1];
	total_size[0]=_size;
	
	/* Buffer */
	cl_mem cl_img = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint) * _size, &img[0], NULL);
	cl_mem cl_total = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint) * 1, &total_size[0], NULL);
	cl_mem cl_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint) * 256 * 3, &ptr[0], NULL);
	if(cl_img == 0 || cl_total == 0 || cl_result == 0) {
		cout << "Can't create OpenCL buffer\n";
		clReleaseMemObject(cl_img);
		clReleaseMemObject(cl_total);
		clReleaseMemObject(cl_result);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		exit(1);
	}

	/* Load kernel */
	FILE *fp;
	const char fileName[] = "./histogram.cl";
	size_t source_size;
	char *source_str;
 
	fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load histograme kernel.\n");	
		exit(1);
	}	

	source_str = (char *)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	/* Compile and Build OpenCL program */	
	const char* source = source_str;
	
	cl_program program = clCreateProgramWithSource(context, 1, &source, 0, 0);
	if(program == 0) {
		cout << "Can't load program\n";
		exit(1);
	}	

	err=clBuildProgram(program, 0, 0, 0, 0, 0);
	
	if(err != CL_SUCCESS) {
		cout << "Can't build program\n";
		exit(1);
	}
	
	if(program == 0) {
		cout << "Can't load or build program\n";
		clReleaseMemObject(cl_img);
		clReleaseMemObject(cl_total);
		clReleaseMemObject(cl_result);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		exit(1);
	}
	
	cl_kernel computeImageRgb = clCreateKernel(program, "computeImageRgb", 0);
	if(computeImageRgb == 0) {
		cout << "Can't load kernel 1" << endl;
		clReleaseProgram(program);
		clReleaseMemObject(cl_img);
		clReleaseMemObject(cl_total);
		clReleaseMemObject(cl_result);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		exit(1);
	}

	/* Setting parameters */
	clSetKernelArg(computeImageRgb, 0, sizeof(cl_mem), &cl_img);
	clSetKernelArg(computeImageRgb, 1, sizeof(cl_mem), &cl_total);
	clSetKernelArg(computeImageRgb, 2, sizeof(cl_mem), &cl_result);

	/* Run */
	size_t work_size = _size;
	err = clEnqueueNDRangeKernel(queue, computeImageRgb, 1, 0, &work_size, 0, 0, 0, 0);
	
	if(err == CL_SUCCESS) {
		err = clEnqueueReadBuffer(queue, cl_result, CL_TRUE, 0, sizeof(unsigned int) * 256 * 3, &ptr[0], 0, 0, 0);
	} else {
		cout << "Can't run kernel 2"<<endl;
	}
	
	/* Result */
	
	if(err != CL_SUCCESS) {		
		cout << "Can't run kernel or read back data"<<"  "<< err<<endl;
	}
		
	clReleaseKernel(computeImageRgb);
	clReleaseProgram(program);
	clReleaseMemObject(cl_img);
	clReleaseMemObject(cl_total);
	clReleaseMemObject(cl_result);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	return ref_histogram_results;
}

int main(int argc, char const *argv[])
{

	unsigned int * histogram_results;
	unsigned int i=0, a, input_size;
	fstream inFile("input", ios_base::in);
	ofstream outFile("0656092.out", ios_base::out);

	inFile >> input_size;
	unsigned int *image = new unsigned int[input_size];
	while( inFile >> a ) {
		image[i++] = a;
	}

	histogram_results = histogram(image, input_size);
	for(unsigned int i = 0; i < 256 * 3; ++i) {
		if (i % 256 == 0 && i != 0)
			outFile << endl;
		outFile << histogram_results[i]<< ' ';
	}

	inFile.close();
	outFile.close();

	return 0;
}
