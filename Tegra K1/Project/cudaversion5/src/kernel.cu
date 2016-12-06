#include "cuda.h"
#include "kernel.cuh"

#define N_PARTICULE 1024
#define MODULO 80

#define TO 1.0f
#define DT 0.001f
#define M 2.0f


__global__ void kernel_updateGalaxy( float4 *p,float4 *vitesse) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j;
	float4 acc;  
	if (i<N_PARTICULE)
	{
		acc=make_float4(0.0f,0.0f,0.0f, 0.0f);
		
		#pragma unroll 4
		for (j=0; j<N_PARTICULE; j++){
			float dx,dy, dz;	
			float dist, coef;
			dx=p[j].x-p[i].x;
			dy=p[j].y-p[i].y;
			dz=p[j].z-p[i].z;	
			dist = sqrtf(dx*dx+dy*dy+dz*dz);
			if (dist < 1.0f)
				coef= vitesse[j].w;
			else
				coef= vitesse[j].w / (dist* dist * dist) ;
			acc.x += dx * coef;
			acc.y += dy * coef;
			acc.z += dz * coef;
		}
		vitesse[i].x+=acc.x*TO* M;
		vitesse[i].y+=acc.y*TO* M;
		vitesse[i].z+=acc.z*TO* M;
	}
	
	
}

__global__ void kernel_updatePos( float4 *p, float4 *vitesse)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<N_PARTICULE)
	{
		p[i].x+=vitesse[i].x*DT;
		p[i].y+=vitesse[i].y*DT;
		p[i].z+=vitesse[i].z*DT;
	}
}

void updateGalaxy( int nblocks, int nthreads, float4 *p,float4 *acceleration) {
	kernel_updateGalaxy<<<nblocks, nthreads>>>( p, acceleration);
	kernel_updatePos<<<nblocks, nthreads>>>(p, acceleration);
}

