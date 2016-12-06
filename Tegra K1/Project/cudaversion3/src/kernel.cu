#include "cuda.h"
#include "particle.h"
#include "kernel.cuh"
#include <math.h>
#include <stdio.h>


__global__ void kernel_updateGalaxy( float *m, float3 *p,float3 *acceleration, float3 *p_out  ) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j;
	float3 acc;  
	if (i<N_PARTICULE)
	{
		acc=make_float3(0.0f,0.0f,0.0f);
		
	//printf("m=%f , px=%f\n", list[i].m, list[i].px);
		for (j=0; j<N_PARTICULE; j++){
			float dx,dy, dz;	
			float dist, coef;
			dx=p[j].x-p[i].x;
			dy=p[j].y-p[i].y;
			dz=p[j].z-p[i].z;	
			dist = sqrtf(dx*dx+dy*dy+dz*dz);
			if (dist < 1.0f)
				dist= 1.0f;
			coef= m[j] / (dist* dist * dist) ;
			acc.x += dx * coef;
			acc.y += dy * coef;
			acc.z += dz * coef;
		}
		acceleration[i].x+=acc.x*TO* M;
		acceleration[i].y+=acc.y*TO* M;
		acceleration[i].z+=acc.z*TO* M;
		p_out[i].x=acceleration[i].x*DT+p[i].x;
		p_out[i].y=acceleration[i].y*DT+p[i].y;
		p_out[i].z=acceleration[i].z*DT+p[i].z;


	}
	
	
}

void updateGalaxy( int nblocks, int nthreads, float *m, float3 *p,float3 *acceleration,  float3 *p_out ) {
	kernel_updateGalaxy<<<nblocks, nthreads>>>(m , p, acceleration, p_out );
}

