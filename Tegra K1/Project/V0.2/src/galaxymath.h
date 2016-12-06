#ifndef __GALAXY_MATH_H__
#define __GALAXY_MATH_H__

	#define number		float
	#define EPS_VALUE	0.00001f
	#define MAX_VALUE	FLT_MAX
	#define ZERO		0.0f
	#define T( f )		f

	#define add( a, b )	a + b
	#define sub( a, b )	a - b
	#define mul( a, b )	a * b
	#define div( a, b )	a / b
	#define sqr( a )	MathSqrtf( a )



typedef struct vec3 {
	number x; 	/*!< x component */
	number y; 	/*!< y component */
	number z; 	/*!< z component */
} vec3_t;


vec3_t	MathVec3Scale		( vec3_t a, number c );
vec3_t	MathVec3Add			( vec3_t a, vec3_t b );
vec3_t	MathVec3Set			( number x, number y, number z );

#endif