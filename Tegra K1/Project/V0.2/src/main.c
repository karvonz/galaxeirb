#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>

#include "GL/glew.h"

#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

#include "text.h"

#define FILENAME "../dubinski.tab"
#define TO 1

static float g_inertia = 0.5f;

static float oldCamPos[] = { 0.0f, 0.0f, -45.0f };
static float oldCamRot[] = { 0.0f, 0.0f, 0.0f };
static float newCamPos[] = { 0.0f, 0.0f, -45.0f };
static float newCamRot[] = { 0.0f, 0.0f, 0.0f };

static bool g_showGrid = true;
static bool g_showAxes = true;

//typedef struct particule particule; 


typedef struct particule_t
{
	float m;
	vec3_t p;
	vec3_t v;
} particule;

vec3_t MathVec3Set( number x, number y, number z ) {
	vec3_t v;
	v.x = x;
	v.y = y;
	v.z = z;
	return v;
}

void DrawPoint( vect3_t point)
{	
	glVertex3f( point.x, point.y, point.z);	
}

vect3_t CalculAcceleration( vect3_t i, vect3_t j, float mj){

	float dist = sqrt((i.x-j.x)*(i.y-j.y)+(i.y-j.y)*(i.z-j.z)+(i.z-j.z)*(i.z-j.z));
	return(MathVec3Set((i.x-j.x)*mj/(dist*dist*dist), (i.y-j.y)*mj/(dist*dist*dist), (i.z-j.z)*mj/(dist*dist*dist)));

}

void UpdateGalaxy(particle *list, vect3_t *acceleration ,int size){

	int i, j;
	for (i=0; i<size; i++){
		acceleration[i]=0.0f;
		for (j=0; j<size; j++){
			if( j !=i )
				acceleration[i] = Add(acceleration[i], CalculAcceleration(list[i].p,list[j].p,list[j].m));
		}
		acceleration[i]=MathVec3Scale(acceleration[i], list[i].m*TO);
	}

	//TODO update particle position
	
}

void DrawGalaxy(particule *listparticule, int size)
{		glPointSize(1.0f);
glBegin( GL_POINTS );
	int i;
	for (i=0; i <16384; i++)
	{
		glColor3f( 1.0f, 0.5f, 0.0f );

	DrawPoint(listparticule[i].p);

	}

	for (i=16384; i <32768; i++)
	{
		glColor3f( 0.0f, 0.5f, 1.0f );

		DrawPoint(listparticule[i].p);

	}

for (i=32768; i <40960; i++)
	{
		glColor3f( 1.0f, 0.0f, 0.0f );

		DrawPoint(listparticule[i].p);

	}


for (i=40960; i <49152; i++)
	{
		glColor3f( 0.0f, 0.0f, 1.0f );

		DrawPoint(listparticule[i].p);

	}



for (i=49152; i <65536; i++)
	{
		glColor3f( 1.0f, 1.0f, 0.0f );

		DrawPoint(listparticule[i].p);
}

for (i=65536; i <size; i++)
	{
		glColor3f( 0.0f, 1.0f, 1.0f );

		DrawPoint(listparticule[i].p);

	}

	glEnd();
}

particule *LoadFile(char *file, int *t)
{
	particule *p = NULL;
	FILE* fichier = NULL;
	int i;
	float px, py, pz, vx, vy, vz;
	fichier = fopen(file, "r");
	if (fichier != NULL)
	{	
		fscanf(fichier, "%d", t) ;
		p = (particule*)malloc(sizeof(particule) * (*t));
		if (p == NULL)
		{
			SDL_Log("Probleme allocation memoire liste particule");
			exit(0);
		}


		for (i=0; i< *t; i++)
		{
			//SDL_Log("Test1");


			fscanf(fichier, "%f %f %f %f %f %f %f", &p[i].m, &px, &py, &pz, &vx, &vy, &p[i].vz);
			p[i].p=MathVect3Set(px,py,pz);		
			p[i].v=MathVect3Set(vx,vy,vz);

//SDL_Log("Test2");
		}
		fclose(fichier);
	}else
{
		SDL_Log( "Probleme d'ouverture %s", file);
}


return p;

}

void DrawGridXZ( float ox, float oy, float oz, int w, int h, float sz ) {
	
	int i;

	glLineWidth( 1.0f );

	glBegin( GL_LINES );

	glColor3f( 0.48f, 0.48f, 0.48f );

	for ( i = 0; i <= h; ++i ) {
		glVertex3f( ox, oy, oz + i * sz );
		glVertex3f( ox + w * sz, oy, oz + i * sz );
	}

	for ( i = 0; i <= h; ++i ) {
		glVertex3f( ox + i * sz, oy, oz );
		glVertex3f( ox + i * sz, oy, oz + h * sz );
	}

	glEnd();

}

void ShowAxes() {

	glLineWidth( 2.0f );

	glBegin( GL_LINES );
	
	glColor3f( 1.0f, 0.0f, 0.0f );
	glVertex3f( 0.0f, 0.0f, 0.0f );
	glVertex3f( 2.0f, 0.0f, 0.0f );

	glColor3f( 0.0f, 1.0f, 0.0f );
	glVertex3f( 0.0f, 0.0f, 0.0f );
	glVertex3f( 0.0f, 2.0f, 0.0f );

	glColor3f( 0.0f, 0.0f, 1.0f );
	glVertex3f( 0.0f, 0.0f, 0.0f );
	glVertex3f( 0.0f, 0.0f, 2.0f );
	
	glEnd();

}

int main( int argc, char ** argv ) {

	SDL_Event event;
	SDL_Window * window;
	SDL_DisplayMode current;
  	
	int width = 640;
	int height = 480;

	bool done = false;

	float mouseOriginX = 0.0f;
	float mouseOriginY = 0.0f;

	float mouseMoveX = 0.0f;
	float mouseMoveY = 0.0f;

	float mouseDeltaX = 0.0f;
	float mouseDeltaY = 0.0f;

	struct timeval begin, end;
	float fps = 0.0;
	char sfps[40] = "FPS: ";

	int tailleFichier =0;


	if ( SDL_Init ( SDL_INIT_EVERYTHING ) < 0 ) {
		printf( "error: unable to init sdl\n" );
		return -1;
	}

	if ( SDL_GetDesktopDisplayMode( 0, &current ) ) {
		printf( "error: unable to get current display mode\n" );
		return -1;
	}

	SDL_GL_SetAttribute( SDL_GL_MULTISAMPLEBUFFERS, 1 );
	SDL_GL_SetAttribute( SDL_GL_MULTISAMPLESAMPLES, 4 );
	
	window = SDL_CreateWindow( "SDL", 	SDL_WINDOWPOS_CENTERED, 
										SDL_WINDOWPOS_CENTERED, 
										width, height, 
										SDL_WINDOW_OPENGL );
  
	SDL_GLContext glWindow = SDL_GL_CreateContext( window );

	GLenum status = glewInit();

	if ( status != GLEW_OK ) {
		printf( "error: unable to init glew\n" );
		return -1;
	}

	if ( ! InitTextRes( "./bin/DroidSans.ttf" ) ) {
		printf( "error: unable to init text resources\n" );
		return -1;
	}

	SDL_GL_SetSwapInterval( 1 );

	particule *list = NULL;
	list = LoadFile(FILENAME, &tailleFichier); 

	//creation du tableu pour le calcul de l'acceleration de chaque particule
	float *particleAcceleration = NULL;
	particleAcceleration = malloc(tailleFichier * sizeof(float) );

	if (particleAcceleration == NULL){
		SDL_Log("Erreur creation tableau acceleration");
		exit(0);
	}

	while ( !done ) {
  		
		int i;

		while ( SDL_PollEvent( &event ) ) {
      
			unsigned int e = event.type;
			
			if ( e == SDL_MOUSEMOTION ) {
				mouseMoveX = event.motion.x;
				mouseMoveY = height - event.motion.y - 1;
			} else if ( e == SDL_KEYDOWN ) {
				if ( event.key.keysym.sym == SDLK_F1 ) {
					g_showGrid = !g_showGrid;
				} else if ( event.key.keysym.sym == SDLK_F2 ) {
					g_showAxes = !g_showAxes;
				} else if ( event.key.keysym.sym == SDLK_ESCAPE ) {
 					done = true;
				}
			}

			if ( e == SDL_QUIT ) {
				printf( "quit\n" );
				done = true;
			}

		}

		mouseDeltaX = mouseMoveX - mouseOriginX;
		mouseDeltaY = mouseMoveY - mouseOriginY;

		if ( SDL_GetMouseState( 0, 0 ) & SDL_BUTTON_LMASK ) {
			oldCamRot[ 0 ] += -mouseDeltaY / 5.0f;
			oldCamRot[ 1 ] += mouseDeltaX / 5.0f;
		}else if ( SDL_GetMouseState( 0, 0 ) & SDL_BUTTON_RMASK ) {
			oldCamPos[ 2 ] += ( mouseDeltaY / 100.0f ) * 0.5 * fabs( oldCamPos[ 2 ] );
			oldCamPos[ 2 ]  = oldCamPos[ 2 ] > -5.0f ? -5.0f : oldCamPos[ 2 ];
		}

		mouseOriginX = mouseMoveX;
		mouseOriginY = mouseMoveY;

		glViewport( 0, 0, width, height );
		glClearColor( 0.2f, 0.2f, 0.2f, 1.0f );
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
		glEnable( GL_BLEND );
		glBlendEquation( GL_FUNC_ADD );
		glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
		glDisable( GL_TEXTURE_2D );
		glEnable( GL_DEPTH_TEST );
		glMatrixMode( GL_PROJECTION );
		glLoadIdentity();
		gluPerspective( 50.0f, (float)width / (float)height, 0.1f, 100000.0f );
		glMatrixMode( GL_MODELVIEW );
		glLoadIdentity();

		for ( i = 0; i < 3; ++i ) {
			newCamPos[ i ] += ( oldCamPos[ i ] - newCamPos[ i ] ) * g_inertia;
			newCamRot[ i ] += ( oldCamRot[ i ] - newCamRot[ i ] ) * g_inertia;
		}

		glTranslatef( newCamPos[0], newCamPos[1], newCamPos[2] );
		glRotatef( newCamRot[0], 1.0f, 0.0f, 0.0f );
		glRotatef( newCamRot[1], 0.0f, 1.0f, 0.0f );
		
		if ( g_showGrid ) {
			DrawGridXZ( -100.0f, 0.0f, -100.0f, 20, 20, 10.0 );
		}

		if ( g_showAxes ) {
			ShowAxes();
		}

		gettimeofday( &begin, NULL );

		//DrawPoint(1.0f, 1.0f, 1.0f);
		//DrawPoint(1.0f, 2.0f, 1.0f);
		//DrawPoint(1.0f, 2.0f, 2.0f);
		// Simulation should be computed here
		DrawGalaxy(list, tailleFichier);
		UpdateGalaxy(list, particleAcceleration ,taille fichier);

		gettimeofday( &end, NULL );

		fps = (float)1.0f / ( ( end.tv_sec - begin.tv_sec ) * 1000000.0f + end.tv_usec - begin.tv_usec) * 1000000.0f;
		sprintf( sfps, "FPS : %.4f", fps );

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluOrtho2D(0, width, 0, height);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		DrawText( 10, height - 20, sfps, TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
		DrawText( 10, 30, "'F1' : show/hide grid", TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
		DrawText( 10, 10, "'F2' : show/hide axes", TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );

		SDL_GL_SwapWindow( window );
		SDL_UpdateWindowSurface( window );
	}

	SDL_GL_DeleteContext( glWindow );
	DestroyTextRes();
	SDL_DestroyWindow( window );
	SDL_Quit();
	free(listparticule);
	return 1;
}
