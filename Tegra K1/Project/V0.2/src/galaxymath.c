#include "galaxymath.h"

vec3_t MathVec3Set( number x, number y, number z ) {
	vec3_t v;
	v.x = x;
	v.y = y;
	v.z = z;
	return v;
}

vec3_t Add( vec3_t a, vec3_t b ) {
	vec3_t r = MathVec3Set( add( a.x, b.x ), add( a.y, b.y ), add( a.z, b.z ) );
	return r;
}

vec3_t MathVec3Scale( vec3_t a, number c ) {
	vec3_t r = MathVec3Set( mul( c, a.x ), mul( c, a.y ), mul( c, a.z ) );
	return r;
}
