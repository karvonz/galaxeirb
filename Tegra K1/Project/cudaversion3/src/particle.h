#define N_PARTICULE 1024
#define MODULO 80

#define TO 1.0f
#define DT 0.01f
#define M 2.0f


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
