

#define MODULO 80
#define NB_PARTICLES (81920/MODULO)
#define TO 1.0f
#define DT 0.05f
#define M 8.0f
//#include "math.h"


float4 bodyBodyInteraction(float4 bi, float4 bj, float4 ai)
{
	float4 r;

	r.x = bj.x - bi.x;
	r.y = bj.y - bi.y;
	r.z = bj.z - bi.z;

	float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;

	float dist = sqrt(distSqr);
	if (dist < 1.0f) return ai;

	float distCube = dist * dist * dist;



	float s = bj.w / distCube;
	//float s = 1.0f / distCube;

	ai.x += r.x * s ;//* ep;
	ai.y += r.y * s ;//* ep;
	ai.z += r.z * s ;//* ep;

	return ai;
}

float4 tile_calculation(float4 myPosition, float4 acc, unsigned int bsize)
{
	 __local float4 *shPosition;

#pragma unroll 8
	for (unsigned int i = 0; i < bsize; i++)
		acc = bodyBodyInteraction(myPosition, shPosition[i], acc);

	return acc;
}

__kernel void galaxeirb(__global float4* pdata, __global float4* pos)
{

	// shared memory
	__local float4 shPosition[1024];

	// index of my body	
	unsigned int x = get_global_id(0);
	unsigned int y = get_global_id(1);
	const unsigned int width = get_local_size(0) * get_local_size(1);
	const unsigned int height = NB_PARTICLES / width;
	unsigned int pLoc = y * width + x;
	unsigned int vLoc = width * height + pLoc;

	// starting index of position array
	//unsigned int start = 0;//( (width * height));

	float4 myPosition = pdata[pLoc];
	float4 myVelocity = pdata[vLoc];

	float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

	unsigned int idx = 0;
	unsigned int loop = ((width * height)) / width;


	for (int i = 0; i < loop; i++)
	{
		idx = get_local_id(1) * get_local_size(0) + get_local_id(0);
		shPosition[idx] = pdata[idx + width * i];

		barrier(CLK_LOCAL_MEM_FENCE);

		acc = tile_calculation(myPosition, acc, width);
		for (unsigned int i = 0; i < width; i++)
			acc = bodyBodyInteraction(myPosition, shPosition[i], acc);

		barrier(CLK_LOCAL_MEM_FENCE);		
	}

	// update velocity with above acc
	myVelocity.x += acc.x * M;// * 2.0f;
	myVelocity.y += acc.y * M;// * 2.0f;
	myVelocity.z += acc.z * M;// * 2.0f;

	// myVelocity.x *= damping;
	// myVelocity.y *= damping;
	// myVelocity.z *= damping;

	// update position
	myPosition.x += myVelocity.x * DT;
	myPosition.y += myVelocity.y * DT;
	myPosition.z += myVelocity.z * DT;

	barrier(CLK_LOCAL_MEM_FENCE);

	// update device memory
	pdata[pLoc] = myPosition;
	pdata[vLoc] = myVelocity;

	// update vbo
	pos[pLoc] = myPosition;


}
