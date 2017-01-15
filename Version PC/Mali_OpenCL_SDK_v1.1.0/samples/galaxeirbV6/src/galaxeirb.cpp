#include "common.h"
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include "esUtil.h"
#include <cstddef>
#include <math.h>
#include <utility>
#include <CL/cl.h>
#include <sys/time.h>

#define MODULO 80
#define NB_PARTICLES (81920/MODULO)
#define TO 0.5f
#define DT 0.01f
#define M 2
#define FILENAME "../dubinski.tab"
#define MOYENNAGE 20

using namespace std;

#define WORKDIMENSION 1


/*
 * This confidential and proprietary software may be used only as
 * authorised by a licensing agreement from ARM Limited
 *   (C) COPYRIGHT 2013 ARM Limited
 *       ALL RIGHTS RESERVED
 * The entire notice above must be reproduced on all authorised
 * copies and copies may only be made to the extent permitted
 * by a licensing agreement from ARM Limited.
 */





bool printProfilingInfo(cl_event event)
{
    cl_ulong queuedTime = 0;
    if (!checkSuccess(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queuedTime, NULL)))
    {
        cerr << "Retrieving CL_PROFILING_COMMAND_QUEUED OpenCL profiling information failed. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    cl_ulong submittedTime = 0;
    if (!checkSuccess(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &submittedTime, NULL)))
    {
        cerr << "Retrieving CL_PROFILING_COMMAND_SUBMIT OpenCL profiling information failed. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    cl_ulong startTime = 0;
    if (!checkSuccess(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL)))
    {
        cerr << "Retrieving CL_PROFILING_COMMAND_START OpenCL profiling information failed. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    cl_ulong endTime = 0;
    if (!checkSuccess(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL)))
    {
        cerr << "Retrieving CL_PROFILING_COMMAND_END OpenCL profiling information failed. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    cout << "Profiling information:\n";
    /* OpenCL returns times in nano seconds. Print out the times in milliseconds (divide by a million). */
    cout << "Queued time: \t" << (submittedTime - queuedTime) / 1000000.0 << "ms\n";
    cout << "Wait time: \t" << (startTime - submittedTime) / 1000000.0 << "ms\n";
    cout << "Run time: \t" << (endTime - startTime) / 1000000.0 << "ms" << endl;

    return true;
}

bool printSupported2DImageFormats(cl_context context)
{
    /* Get the number of supported image formats in order to allocate the correct amount of memory. */
    cl_uint numberOfImageFormats;
    if (!checkSuccess(clGetSupportedImageFormats(context, 0, CL_MEM_OBJECT_IMAGE2D, 0, NULL, &numberOfImageFormats)))
    {
        cerr << "Getting the number of supported 2D image formats failed. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    /* Get the list of supported image formats. */
    cl_image_format* imageFormats = new cl_image_format[numberOfImageFormats];
    if (!checkSuccess(clGetSupportedImageFormats(context, 0, CL_MEM_OBJECT_IMAGE3D, numberOfImageFormats, imageFormats, NULL)))
    {
        cerr << "Getting the list of supported 2D image formats failed. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    cout << numberOfImageFormats << " Image formats supported";

    if (numberOfImageFormats > 0)
    {
        cout << " (channel order, channel data type):" << endl;
    }
    else
    {
        cout << "." << endl;
    }

    for (unsigned int i = 0; i < numberOfImageFormats; i++)
    {
        cout << imageChannelOrderToString(imageFormats[i].image_channel_order) << ", " << imageChannelDataTypeToString(imageFormats[i].image_channel_data_type) << endl;
    }

    delete[] imageFormats;

    return true;
}

string imageChannelOrderToString(cl_channel_order channelOrder)
{
    switch (channelOrder)
    {
        case CL_R:
            return "CL_R";
        case CL_A:
            return "CL_A";
        case CL_RG:
             return "CL_RG";
        case CL_RA:
             return "CL_RA";
        case CL_RGB:
            return "CL_RGB";
        case CL_RGBA:
            return "CL_RGBA";
        case CL_BGRA:
            return "CL_BGRA";
        case CL_ARGB:
            return "CL_ARGB";
        case CL_INTENSITY:
            return "CL_INTENSITY";
        case CL_LUMINANCE:
            return "CL_LUMINANCE";
        case CL_Rx:
            return "CL_Rx";
        case CL_RGx:
            return "CL_RGx";
        case CL_RGBx:
            return "CL_RGBx";
        default:
            return "Unknown image channel order";
    }
}

string imageChannelDataTypeToString(cl_channel_type channelDataType)
{
    switch (channelDataType)
    {
        case CL_SNORM_INT8:
            return "CL_SNORM_INT8";
        case CL_SNORM_INT16:
            return "CL_SNORM_INT16";
        case CL_UNORM_INT8:
            return "CL_UNORM_INT8";
        case CL_UNORM_INT16:
            return "CL_UNORM_INT16";
        case CL_UNORM_SHORT_565:
            return "CL_UNORM_SHORT_565";
        case CL_UNORM_SHORT_555:
            return "CL_UNORM_SHORT_555";
        case CL_UNORM_INT_101010:
            return "CL_UNORM_INT_101010";
        case CL_SIGNED_INT8:
            return "CL_SIGNED_INT8";
        case CL_SIGNED_INT16:
            return "CL_SIGNED_INT16";
        case CL_SIGNED_INT32:
            return "CL_SIGNED_INT32";
        case CL_UNSIGNED_INT8:
            return "CL_UNSIGNED_INT8";
        case CL_UNSIGNED_INT16:
            return "CL_UNSIGNED_INT16";
        case CL_UNSIGNED_INT32:
            return "CL_UNSIGNED_INT32";
        case CL_HALF_FLOAT:
            return "CL_HALF_FLOAT";
        case CL_FLOAT:
            return "CL_FLOAT";
        default:
            return "Unknown image channel data type";
    }
}

bool cleanUpOpenCL(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel, cl_mem* memoryObjects, int numberOfMemoryObjects)
{
    bool returnValue = true;
    if (context != 0)
    {
        if (!checkSuccess(clReleaseContext(context)))
        {
            cerr << "Releasing the OpenCL context failed. " << __FILE__ << ":"<< __LINE__ << endl;
            returnValue = false;
        }
    }

    if (commandQueue != 0)
    {
        if (!checkSuccess(clReleaseCommandQueue(commandQueue)))
        {
            cerr << "Releasing the OpenCL command queue failed. " << __FILE__ << ":"<< __LINE__ << endl;
            returnValue = false;
        }
    }

    if (kernel != 0)
    {
        if (!checkSuccess(clReleaseKernel(kernel)))
        {
            cerr << "Releasing the OpenCL kernel failed. " << __FILE__ << ":"<< __LINE__ << endl;
            returnValue = false;
        }
    }

    if (program != 0)
    {
        if (!checkSuccess(clReleaseProgram(program)))
        {
            cerr << "Releasing the OpenCL program failed. " << __FILE__ << ":"<< __LINE__ << endl;
            returnValue = false;
        }
    }

    for (int index = 0; index < numberOfMemoryObjects; index++)
    {
        if (memoryObjects[index] != 0)
        {
            if (!checkSuccess(clReleaseMemObject(memoryObjects[index])))
            {
                cerr << "Releasing the OpenCL memory object " << index << " failed. " << __FILE__ << ":"<< __LINE__ << endl;
                returnValue = false;
            }
        }
    }

    return returnValue;
}

bool createContext(cl_context* context)
{
    cl_int errorNumber = 0;
    cl_uint numberOfPlatforms = 0;
    cl_platform_id firstPlatformID = 0;

    /* Retrieve a single platform ID. */
    if (!checkSuccess(clGetPlatformIDs(1, &firstPlatformID, &numberOfPlatforms)))
    {
        cerr << "Retrieving OpenCL platforms failed. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    if (numberOfPlatforms <= 0)
    {
        cerr << "No OpenCL platforms found. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    /* Get a context with a GPU device from the platform found above. */
    cl_context_properties contextProperties [] = {CL_CONTEXT_PLATFORM, (cl_context_properties)firstPlatformID, 0};
    *context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errorNumber);
    if (!checkSuccess(errorNumber))
    {
        cerr << "Creating an OpenCL context failed. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    return true;
}

bool createCommandQueue(cl_context context, cl_command_queue* commandQueue, cl_device_id* device)
{
    cl_int errorNumber = 0;
    cl_device_id* devices = NULL;
    size_t deviceBufferSize = -1;

    /* Retrieve the size of the buffer needed to contain information about the devices in this OpenCL context. */
    if (!checkSuccess(clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize)))
    {
        cerr << "Failed to get OpenCL context information. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    if(deviceBufferSize == 0)
    {
        cerr << "No OpenCL devices found. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    /* Retrieve the list of devices available in this context. */
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    if (!checkSuccess(clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL)))
    {
        cerr << "Failed to get the OpenCL context information. " << __FILE__ << ":"<< __LINE__ << endl;
        delete [] devices;
        return false;
    }

    /* Use the first available device in this context. */
    *device = devices[0];
    delete [] devices;

    /* Set up the command queue with the selected device. */
    *commandQueue = clCreateCommandQueue(context, *device, CL_QUEUE_PROFILING_ENABLE, &errorNumber);
    if (!checkSuccess(errorNumber))
    {
        cerr << "Failed to create the OpenCL command queue. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    return true;
}

bool createProgram(cl_context context, cl_device_id device, string filename, cl_program* program)
{
    cl_int errorNumber = 0;
    ifstream kernelFile(filename.c_str(), ios::in);

    if(!kernelFile.is_open())
    {
        cerr << "Unable to open " << filename << ". " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    /*
     * Read the kernel file into an output stream.
     * Convert this into a char array for passing to OpenCL.
     */
    ostringstream outputStringStream;
    outputStringStream << kernelFile.rdbuf();
    string srcStdStr = outputStringStream.str();
    const char* charSource = srcStdStr.c_str();

    *program = clCreateProgramWithSource(context, 1, &charSource, NULL, &errorNumber);
    if (!checkSuccess(errorNumber) || program == NULL)
    {
        cerr << "Failed to create OpenCL program. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    /* Try to build the OpenCL program. */
    bool buildSuccess = checkSuccess(clBuildProgram(*program, 0, NULL, NULL, NULL, NULL));

    /* Get the size of the build log. */
    size_t logSize = 0;
    clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

    /*
     * If the build succeeds with no log, an empty string is returned (logSize = 1),
     * we only want to print the message if it has some content (logSize > 1).
     */
    if (logSize > 1)
    {
        char* log = new char[logSize];
        clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);

        string* stringChars = new string(log, logSize);
        cerr << "Build log:\n " << *stringChars << endl;

        delete[] log;
        delete stringChars;
    }

    if (!buildSuccess)
    {
        clReleaseProgram(*program);
        cerr << "Failed to build OpenCL program. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    return true;
}

inline bool checkSuccess(cl_int errorNumber)
{
    if (errorNumber != CL_SUCCESS)
    {
        cerr << "OpenCL error: " << errorNumberToString(errorNumber) << endl;
        return false;
    }
    return true;
}

string errorNumberToString(cl_int errorNumber)
{
    switch (errorNumber)
    {
        case CL_SUCCESS:
            return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:
            return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:
            return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:
            return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:
            return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:
            return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:
            return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:
            return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:
            return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:
            return "CL_MAP_FAILURE";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_INVALID_VALUE:
            return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:
            return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:
            return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:
            return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:
            return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:
            return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:
            return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:
            return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:
            return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:
            return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:
            return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:
            return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:
            return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:
            return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:
            return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:
            return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:
            return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:
            return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:
            return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:
            return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:
            return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:
            return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:
            return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:
            return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:
            return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:
            return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:
            return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:
            return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:
            return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:
            return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:
            return "CL_INVALID_MIP_LEVEL";
        default:
            return "Unknown error";
    }
}

bool isExtensionSupported(cl_device_id device, string extension)
{
    if (extension.empty())
    {
        return false;
    }

    /* First find out how large the ouput of the OpenCL device query will be. */
    size_t extensionsReturnSize = 0;
    if (!checkSuccess(clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, NULL, &extensionsReturnSize)))
    {
        cerr << "Failed to get return size from clGetDeviceInfo for parameter CL_DEVICE_EXTENSIONS. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    /* Allocate enough memory for the output. */
    char* extensions = new char[extensionsReturnSize];

    /* Get the list of all extensions supported. */
    if (!checkSuccess(clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, extensionsReturnSize, extensions, NULL)))
    {
        cerr << "Failed to get data from clGetDeviceInfo for parameter CL_DEVICE_EXTENSIONS. " << __FILE__ << ":"<< __LINE__ << endl;
        delete [] extensions;
        return false;
    }

    /* See if the requested extension is in the list. */
    string* extensionsString = new string(extensions);
    bool returnResult = false;
    if (extensionsString->find(extension) != string::npos)
    {
        returnResult = true;
    }

    delete [] extensions;
    delete extensionsString;

    return returnResult;
}


cl_context context = 0;
cl_command_queue commandQueue = 0;
cl_program program = 0;
cl_device_id device = 0;
cl_kernel kernel = 0;
cl_kernel kernel2 = 0;
const unsigned int numberOfMemoryObjects = 2;
cl_mem memoryObjects[numberOfMemoryObjects] = {0, 0};
cl_int errorNumber;
size_t bufferSize = NB_PARTICLES * sizeof(cl_float4);

bool createMemoryObjectsSuccess = true;
cl_float4* popencl;
cl_float4* accopencl;

	float fps = 0.0f;
	float fpstemp= 0.0f;
	int counter = 0;
	float fpsmax = -10.0f;
	float fpsmin = 10000.0f;


	struct timeval begin, end;





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

int tailleFichier =0;
GLfloat vVertices[4*NB_PARTICLES];
particule *list = NULL;




void setColor (){
	

for (int i=0; i <NB_PARTICLES; i++)
	{

		if (i*MODULO > 16384 && i*MODULO<32768)
			vVertices[i*4+3] = 0.0f;
		else if  (i*MODULO >= 40961 && i*MODULO < 49152) 
			vVertices[i*4+3] = 1.0f;
		else if ( i*MODULO >= 65536)
			vVertices[i*4+3] = 2.0f;
		else if (i*MODULO < 16384)
			vVertices[i*4+3] = 3.0f;
		else if  (i*MODULO>=32768 && i*MODULO<40961) 
			vVertices[i*4+3] = 4.0f;
		else if  (i*MODULO>=49152 && i*MODULO<65536) 
			vVertices[i*4+3] = 5.0f;
}




}



void UpdateVertices (){
	int i;
	int j=0;

	for(i=0 ; i<NB_PARTICLES*4 ; i=i+4  )
	{


		vVertices[i]=popencl[j].x/64.0f;
		vVertices[i+1]=popencl[j].y/64.0f;
		vVertices[i+2]=popencl[j].z/64.0f;
		j++;
	}
}


particule *LoadFile(char *file, int *t)
{
	particule *p = NULL, buf;
	FILE* fichier = NULL;
	int i,j=0;
	fichier = fopen(file, "r");
	if (fichier != NULL)
	{	
		fscanf(fichier, "%d", t) ;
		p = (particule*)malloc(sizeof(particule) * (*t));
		if (p == NULL)
		{
			printf("Probleme allocation memoire liste particule");
			exit(0);
		}

		for (i=0; i< *t; i++)
		{
			if ((i %MODULO) == 0)
				fscanf(fichier, "%f %f %f %f %f %f %f", &p[j].m, &p[j].px, &p[j].py, &p[j].pz, &p[j].vx, &p[j].vy, &p[j++].vz);

			else {
				fscanf(fichier, "%f %f %f %f %f %f %f", &buf.m, &buf.px, &buf.py, &buf.pz, &buf.vx, &buf.vy, &buf.vz);
			}
		}
		fclose(fichier);
	}else
	{
		printf( "Probleme d'ouverture %s\n", file);
	}

	*t=j;

	return p;

}

typedef struct
{
	// Handle to a program object
	GLuint programObject;

} UserData;

///
// Create a shader object, load the shader source, and
// compile the shader.
//
GLuint LoadShader ( GLenum type, const char *shaderSrc )
{
	GLuint shader;
	GLint compiled;

	// Create the shader object
	shader = glCreateShader ( type );

	if ( shader == 0 )
		return 0;

	// Load the shader source
	glShaderSource ( shader, 1, &shaderSrc, NULL );

	// Compile the shader
	glCompileShader ( shader );

	// Check the compile status
	glGetShaderiv ( shader, GL_COMPILE_STATUS, &compiled );

	if ( !compiled ) 
	{
		GLint infoLen = 0;

		glGetShaderiv ( shader, GL_INFO_LOG_LENGTH, &infoLen );

		if ( infoLen > 1 )
		{
			char* infoLog = (char*) malloc (sizeof(char) * infoLen );

			glGetShaderInfoLog ( shader, infoLen, NULL, infoLog );
			esLogMessage ( "Error compiling shader:\n%s\n", infoLog );            

			free ( infoLog );
		}

		glDeleteShader ( shader );
		return 0;
	}

	return shader;

}

///
// Initialize the shader and program object
//
int Init ( ESContext *esContext )
{
	esContext->userData = (UserData*) malloc(sizeof(UserData));

	UserData *userData = (UserData*) esContext->userData;
	GLbyte vShaderStr[] =  
		"attribute vec4 vPosition;    \n"
		"varying vec4 v_color; //output vertex color \n"
		"void main()                  \n"
		"{                            \n"
		"   gl_Position.x = vPosition.x;  \n"
		"   gl_Position.y = vPosition.y;  \n"
		"   gl_Position.z = vPosition.z;  \n"
		"   gl_Position.w = 1.0;  \n"
	"	if (vPosition.w == 0.0) \n"
	"		v_color= vec4 (0.0, 0.5, 1.0 , 1.0); \n"
	"	else {                                        \n"
		"	if (vPosition.w == 1.0) \n"
	"		v_color= vec4 (0.0, 0.0, 1.0, 1.0 ); \n"
"	else {                                        \n"
		"	if (vPosition.w == 2.0) \n"
	"		v_color= vec4 (0.0, 0.2, 1.0, 1.0 ); \n"
	"	else {                                        \n"
		"	if (vPosition.w == 3.0) \n"
	"		v_color= vec4 (1.0, 0.5, 0.0, 1.0 ); \n"
	"	else {                                        \n"
		"	if (vPosition.w == 4.0) \n"
	"		v_color= vec4 (1.0, 0.0, 0.0, 1.0 ); \n"
			"else \n"
	"		v_color= vec4 (1.0, 0.2, 1.0, 1.0 ); \n"
	" }}}} \n"
		"}                            \n";




	GLbyte fShaderStr[] =  
		"precision mediump float;\n"\
		"varying vec4 v_color; \n"
		"void main()                                  \n"
		"{  \n"
			"	gl_FragColor = v_color;\n"	                                     
			"}                                            \n";

	GLuint vertexShader;
	GLuint fragmentShader;
	GLuint programObject;
	GLint linked;

	// Load the vertex/fragment shaders
	vertexShader =  LoadShader ( GL_VERTEX_SHADER, ( char*) vShaderStr );
	fragmentShader = LoadShader ( GL_FRAGMENT_SHADER, (char *) fShaderStr );


	printf( "%d %d\n",  vertexShader, fragmentShader);

	// Create the program object
	programObject = glCreateProgram ( );

	if ( programObject == 0 )
		return 0;


	glAttachShader ( programObject, vertexShader );
	glAttachShader ( programObject, fragmentShader );

	// Bind vPosition to attribute 0   
	glBindAttribLocation ( programObject, 0, "vPosition" );

	// Link the program
	glLinkProgram ( programObject );



	// Check the link status
	glGetProgramiv ( programObject, GL_LINK_STATUS, &linked );

	if ( !linked ) 
	{
		GLint infoLen = 0;

		glGetProgramiv ( programObject, GL_INFO_LOG_LENGTH, &infoLen );

		if ( infoLen > 1 )
		{
			char* infoLog = (char *) malloc (sizeof(char) * infoLen );

			glGetProgramInfoLog ( programObject, infoLen, NULL, infoLog );
			esLogMessage ( "Error linking program:\n%s\n", infoLog );            

			free ( infoLog );
		}

		glDeleteProgram ( programObject );
		return GL_FALSE;
	}

	// Store the program object
	userData->programObject = programObject;

	glClearColor ( 0.0f, 0.0f, 0.0f, 0.0f );

	
	return GL_TRUE;
}



/////////////////////////////////////////////////////////////////////
//				DRAW				   //
/////////////////////////////////////////////////////////////////////

void Draw ( ESContext *esContext )
{

size_t globalWorksize[1] = {NB_PARTICLES};


	gettimeofday( &begin, NULL );

	UserData *userData = (UserData*) esContext->userData;



	/* Setup the kernel arguments.  */
	bool setKernelArgumentsSuccess = true;
	setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, 0, sizeof(cl_mem), &memoryObjects[0]));
	setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, 1, sizeof(cl_mem), &memoryObjects[1]));

	if (!setKernelArgumentsSuccess)
	{
		cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
		cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
		return ;
	}

	
////////////////////////////////
//  	Lancement Kernel      //
////////////////////////////////
	cl_event event = 0;
	/* Enqueue the kernel */
	if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorksize, NULL, 0, NULL, &event)))
	{
		cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
		cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
		return ;
	}

	/* Wait for kernel execution completion. */
	if (!checkSuccess(clFinish(commandQueue)))
	{
		cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
		cerr << "Failed waiting for kernel execution to finish. " << __FILE__ << ":"<< __LINE__ << endl;
		return ;
	}

	/* Print the profiling information for the event. */
	// printProfilingInfo(event);
	/* Release the event object. */
	if (!checkSuccess(clReleaseEvent(event)))
	{
		cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
		cerr << "Failed releasing the event object. " << __FILE__ << ":"<< __LINE__ << endl;
		return ;
	}

////////////////////////////////
//  	   Fin Kernel1        //
////////////////////////////////

////////////////////////////////
//     	  Prepa Kernel2       //
////////////////////////////////

/* Setup the kernel arguments.  */
	setKernelArgumentsSuccess = true;
	setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel2, 0, sizeof(cl_mem), &memoryObjects[0]));
	setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel2, 1, sizeof(cl_mem), &memoryObjects[1]));

	if (!setKernelArgumentsSuccess)
	{
		cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
		cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
		return ;
	}
////////////////////////////////
//  	Lancement Kernel2     //
////////////////////////////////
	 event = 0;
	/* Enqueue the kernel */
	if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernel2, 1, NULL, globalWorksize, NULL, 0, NULL, &event)))
	{
		cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
		cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
		return ;
	}

	/* Wait for kernel execution completion. */
	if (!checkSuccess(clFinish(commandQueue)))
	{
		cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
		cerr << "Failed waiting for kernel execution to finish. " << __FILE__ << ":"<< __LINE__ << endl;
		return ;
	}

	/* Print the profiling information for the event. */
	// printProfilingInfo(event);
	/* Release the event object. */
	if (!checkSuccess(clReleaseEvent(event)))
	{
		cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
		cerr << "Failed releasing the event object. " << __FILE__ << ":"<< __LINE__ << endl;
		return ;
	}
////////////////////////////////
//  	   Fin Kernel2        //
////////////////////////////////



	/* Get a pointer to the output data. */
	popencl = (cl_float4*)clEnqueueMapBuffer(commandQueue, memoryObjects[0], CL_TRUE, CL_MAP_READ, 0, bufferSize, 0, NULL, NULL, &errorNumber);
	if (!checkSuccess(errorNumber))
	{
		cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
		cerr << "Failed to map buffer. " << __FILE__ << ":"<< __LINE__ << endl;
		return ;
	}
/*
//a remplacer par appel direct au popencl pour eviter de réecrire
	for(int i=0; i<NB_PARTICLES; i++)
	{
		list[i].px=popencl[i].x;
		list[i].py=popencl[i].y;
		list[i].pz=popencl[i].z;

	} */

		UpdateVertices();
gettimeofday( &end, NULL );

fps = (float)1.0f / ( ( end.tv_sec - begin.tv_sec ) * 1000000.0f + end.tv_usec - begin.tv_usec) * 1000000.0f;

		counter++;
		fpstemp+=fps;

		if (counter == MOYENNAGE-1)
		{
			counter=0;
			printf( "FPS : %.0f\n", fpstemp/(float)MOYENNAGE );
			fpstemp=0.0f;
			
		}
//liberation mémoire

if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[0], popencl, 0, NULL, NULL)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
       return ;
    }


//Fin opencl




	// Set the viewport
	glViewport ( 0, 0, esContext->width, esContext->height );

	// Clear the color buffer
	glClear ( GL_COLOR_BUFFER_BIT );

	// Use the program object
	glUseProgram ( userData->programObject );

	// Load the vertex data
	glVertexAttribPointer ( 0, 4, GL_FLOAT, GL_FALSE, 0,vVertices );
	glEnableVertexAttribArray ( 0 );

	glDrawArrays ( GL_POINTS, 0, NB_PARTICLES );
}



/////////////////////////////////////////////////////////////////////
//				MAIN				   //
/////////////////////////////////////////////////////////////////////



int main ( int argc, char *argv[] )
{
	int i;

	list = LoadFile(FILENAME, &tailleFichier); 

	setColor();
	//UpdateVertices();


////////////////////////////////
//  OPENCL  CONFIGURATION     //
////////////////////////////////


	if (!createContext(&context))
	{
		cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
		cerr << "Failed to create an OpenCL context. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}

	if (!createCommandQueue(context, &commandQueue, &device))
	{
		cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
		cerr << "Failed to create the OpenCL command queue. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}

	if (!createProgram(context, device, "assets/galaxeirb.cl", &program))
	{
		cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
		cerr << "Failed to create OpenCL program." << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}
////////////////////////////////
// 	Creation Kernel       //
////////////////////////////////

	kernel = clCreateKernel(program, "galaxeirb", &errorNumber);
	if (!checkSuccess(errorNumber))
	{
		cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
		cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}


//deuxieme kernel
	kernel2 = clCreateKernel(program, "kernel_updatePos", &errorNumber);
	if (!checkSuccess(errorNumber))
	{
		cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
		cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}


////////////////////////////////
// Initialize MEMORY from CPU //
////////////////////////////////


	memoryObjects[0] = clCreateBuffer(context,  CL_MEM_READ_WRITE  | CL_MEM_ALLOC_HOST_PTR, bufferSize, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	memoryObjects[1] = clCreateBuffer(context,  CL_MEM_READ_WRITE  | CL_MEM_ALLOC_HOST_PTR, bufferSize, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	
	if (!createMemoryObjectsSuccess)
	{
		cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
		cerr << "Failed to create OpenCL buffers. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}




	popencl = (cl_float4*)clEnqueueMapBuffer(commandQueue, memoryObjects[0], CL_TRUE, CL_MAP_WRITE, 0, bufferSize, 0, NULL, NULL, &errorNumber);

	if (!checkSuccess(errorNumber)) //test si le map sont successful
	{
		cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
		cerr << "Mapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}
	
	accopencl = (cl_float4*)clEnqueueMapBuffer(commandQueue, memoryObjects[1], CL_TRUE, CL_MAP_WRITE, 0, bufferSize, 0, NULL, NULL, &errorNumber);



	if (!checkSuccess(errorNumber)) //test si le map sont successful
	{
		cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
		cerr << "Mapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}


//mtn j'écris dans la mémoire en utilisant les pointeurs sur la shared memory
	for (int i = 0; i < NB_PARTICLES; i++)
	{
		popencl[i].x   = list[i].px;
		popencl[i].y   = list[i].py;
		popencl[i].z   = list[i].pz;
		popencl[i].w   = list[i].m;
		accopencl[i].x = list[i].vx;
		accopencl[i].y = list[i].vy;
		accopencl[i].z = list[i].vz;
		accopencl[i].w = list[i].m;
		
	}
	

//Libération de la mémoire côté CPU pour l'utiliser sur le GPU

    if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[0], popencl, 0, NULL, NULL)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }
    if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[1], accopencl, 0, NULL, NULL)))
    {
       cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
       return 1;
    }



////////////////////////////////
// 	Affichage Graphique   //
////////////////////////////////



	ESContext esContext;
	UserData  userData;

	esInitContext ( &esContext );
	esContext.userData = &userData;

	esCreateWindow ( &esContext, "GalaxEirb", 640, 480, ES_WINDOW_RGB );

	if ( !Init ( &esContext ) )
		return 0;



	esRegisterDrawFunc ( &esContext, Draw );

	esMainLoop ( &esContext );

}
