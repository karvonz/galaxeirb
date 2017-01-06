#ifndef _GALAXY_KERNEL_H_
#define _GALAXY_KERNEL_H_

#include "cuda.h"
#include "kernel.cuh"
#include <math.h>
#include <stdio.h>

#define BSIZE 128
#define softeningSquared 0.01f		// original plumer softener is 0.025. here the value is square of it.
#define damping 1.0f				// 0.999f
#define ep 0.67f						// 0.5f
#define DT 0.001f
#define M 2.0f


__device__ float4 bodyBodyInteraction(float4 bi, float4 bj, float4 ai)
{
	float4 r;

	r.x = bj.x - bi.x;
	r.y = bj.y - bi.y;
	r.z = bj.z - bi.z;

	float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;

	float dist = sqrtf(distSqr);
	if (dist < 1.0f) return ai;

	float distCube = dist * dist * dist;



	float s = bj.w / distCube;
	//float s = 1.0f / distCube;

	ai.x += r.x * s ;//* ep;
	ai.y += r.y * s ;//* ep;
	ai.z += r.z * s ;//* ep;

	return ai;
}

__device__ float4 tile_calculation(float4 myPosition, float4 acc)
{
	extern __shared__ float4 shPosition[];

#pragma unroll 8
	for (unsigned int i = 0; i < BSIZE; i++)
		acc = bodyBodyInteraction(myPosition, shPosition[i], acc);

	return acc;
}

__global__ void galaxyKernel(float4* pos, float4 * pdata, unsigned int width, unsigned int height)
{
	//printf("test\n");	

	// shared memory
	extern __shared__ float4 shPosition[];

	// index of my body	
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;


	unsigned int pLoc = y * width + x;
	unsigned int vLoc = width * height + pLoc;

	// starting index of position array
	unsigned int start = 0;//( (width * height));

	float4 myPosition = pdata[pLoc];
	float4 myVelocity = pdata[vLoc];

	float4 acc = {0.0f, 0.0f, 0.0f};

	unsigned int idx = 0;
	unsigned int loop = ((width * height)) / BSIZE;
	for (int i = 0; i < loop; i++)
	{
		idx = threadIdx.y * blockDim.x + threadIdx.x;
		shPosition[idx] = pdata[idx + start + BSIZE * i];

		__syncthreads();

		acc = tile_calculation(myPosition, acc);

		__syncthreads();		
	}

	// update velocity with above acc
	myVelocity.x += acc.x * M* damping;// * 2.0f;
	myVelocity.y += acc.y * M* damping;// * 2.0f;
	myVelocity.z += acc.z * M* damping;// * 2.0f;

	// myVelocity.x *= damping;
	// myVelocity.y *= damping;
	// myVelocity.z *= damping;

	// update position
	myPosition.x += myVelocity.x * DT;
	myPosition.y += myVelocity.y * DT;
	myPosition.z += myVelocity.z * DT;

	__syncthreads();

	// update device memory
	pdata[pLoc] = myPosition;
	pdata[vLoc] = myVelocity;

	// update vbo
	pos[pLoc] = myPosition;


}

//extern "C" 
void cudaComputeGalaxy(float4* pos, float4 * pdata, int width, int height)
{
	dim3 block(16, 8, 1);
	dim3 grid(width / block.x, height / block.y, 1);
	int sharedMemSize = BSIZE * sizeof(float4);

	galaxyKernel<<<grid, block, sharedMemSize>>>(pos, pdata, width, height);
}

#endif

