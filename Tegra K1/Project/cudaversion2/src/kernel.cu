#include "cuda.h"
#include "particle.h"
#include "kernel.cuh"
#include <math.h>
#include <stdio.h>


__global__ void kernel_updateGalaxy( float *m, float3 *p,float3 *acceleration  ) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j;
	if (i<N_PARTICULE)
	{
		acceleration[i]=make_float3(0.0f,0.0f,0.0f);
		
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
			acceleration[i].x += dx * coef;
			acceleration[i].y += dy * coef;
			acceleration[i].z += dz * coef;
		}
	}
}

void updateGalaxy( int nblocks, int nthreads, float *m, float3 *p,float3 *acceleration  ) {
	kernel_updateGalaxy<<<nblocks, nthreads>>>(m , p, acceleration );
}