

#define MODULO 80
#define NB_PARTICLES (81920/MODULO)
#define TO 1.0f
#define DT 0.01f
#define M 2.0f
//#include "math.h"



__kernel void galaxeirb(__global float4* p, __global float4* vitesse)
{
	 int i = get_global_id(0); 
int j;
	float4 acc;  
	if (i<NB_PARTICLES)
	{
		float4 acc;
		acc.x=0.0f;
		acc.y=0.0f;
		acc.z=0.0f;
		acc.w=0.0f;
		

		for (j=0; j<NB_PARTICLES; j++){
			float dx,dy, dz;	
			float dist, coef;
			dx=p[j].x-p[i].x;
			dy=p[j].y-p[i].y;
			dz=p[j].z-p[i].z;	
			dist = sqrt(dx*dx+dy*dy+dz*dz);
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



__kernel void kernel_updatePos(__global float4 *p,__global float4 *vitesse)
{
	int i = get_global_id(0);
	if (i<NB_PARTICLES)
	{
		p[i].x+=vitesse[i].x*DT;
		p[i].y+=vitesse[i].y*DT;
		p[i].z+=vitesse[i].z*DT;
	}
}
