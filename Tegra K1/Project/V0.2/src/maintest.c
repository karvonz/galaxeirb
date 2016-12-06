#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>

typedef struct particule_t
{
	float m;
	float px;
	float py;
	float pz;
	float vx;
	float vy;
	float vz;
} particule;


void LoadFile( int t, particule *p)
{
		p = (particule*)malloc(sizeof(particule) * (t));
}


int main( int argc, char ** argv ) {

	int tailleFichier =12;
	particule *list = NULL;
	LoadFile(list); 
	
	return 1;
}

