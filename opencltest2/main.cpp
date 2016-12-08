
  #define __CL_ENABLE_EXCEPTIONS
  
  #if defined(__APPLE__) || defined(__MACOSX)
  #include <OpenCL/cl.hpp>
  #else
  #include <CL/cl.hpp>
  #endif
  #include <cstdio>
  #include <cstdlib>
  #include <iostream>
  
   const char * helloStr  = "__kernel void "
                            "hello(void) "
                            "{ "
                            "  "
                            "} ";
  
   int  main(void)
   {

cl_uint platformIdCount = 0;
clGetPlatformIDs (0, NULL, &platformIdCount);

std::vector<cl_platform_id> platformIds (platformIdCount);
clGetPlatformIDs (platformIdCount, platformIds.data (), NULL);


cl_uint deviceIdCount = 0;
clGetDeviceIDs (platformIds [0], CL_DEVICE_TYPE_ALL, 0, NULL,
    &deviceIdCount);
std::vector<cl_device_id> deviceIds (deviceIdCount);
clGetDeviceIDs (platformIds [0], CL_DEVICE_TYPE_ALL, deviceIdCount,
    deviceIds.data (), NULL);


	/*//get platform ID
	cl_platform_id platform;
	clGetPlatformIDs(1, &platform, NULL);
	
	//get first GPU device
	cl_device_id device;
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	printf("test\n");
	//create context
	cl_context context;
	context=clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
*/

	printf("test\n");
     /* cl_int err = CL_SUCCESS;
      try {
 	

	

        std::vector<cl::Platform> platforms;
      
	  cl::Platform::get(&platforms);
	std::cout<<"test"<< std::endl;
        if (platforms.size() == 0) {
            std::cout << "Platform size 0\n";
            return -1;
        }
 
        cl_context_properties properties[] = 
           { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
        cl::Context context(CL_DEVICE_TYPE_CPU, properties); 
  
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
  
        cl::Program::Sources source(1,
            std::make_pair(helloStr,strlen(helloStr)));
        cl::Program program_ = cl::Program(context, source);
        program_.build(devices);
  
        cl::Kernel kernel(program_, "hello", &err);
  
        cl::Event event;
        cl::CommandQueue queue(context, devices[0], 0, &err);
        queue.enqueueNDRangeKernel(
            kernel, 
            cl::NullRange, 
            cl::NDRange(4,4),
            cl::NullRange,
            NULL,
            &event); 
  
        event.wait();
      }
      catch (cl::Error err) {
         std::cerr 
            << "ERROR: "
            << err.what()
            << "("
            << err.err()
            << ")"
            << std::endl;
      }*/
  
     return EXIT_SUCCESS;
   }
  
