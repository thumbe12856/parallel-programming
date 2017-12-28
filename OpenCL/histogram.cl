#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
__kernel void computeImageRgb(__global const unsigned int* img, __global const unsigned int* total, __global unsigned int* result)
{
	unsigned int id = get_global_id(0);
	unsigned int offset=( id % 3) * 256;
	unsigned int index = img[id];
	atom_add(&result[index + offset],1);
}
