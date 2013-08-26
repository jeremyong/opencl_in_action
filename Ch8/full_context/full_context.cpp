#define _CRT_SECURE_NO_WARNINGS
#define __CL_ENABLE_EXCEPTIONS
#define __NO_STD_STRING

#include <iostream>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

using namespace std;

int main(void) {
   
   vector<cl::Platform> platforms;
   vector<cl::Device> platformDevices, ctxDevices;
   cl::string device_name;
   cl_uint i;
   try {
      // Access all devices in first platform
      cl::Platform::get(&platforms);
      platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &platformDevices);
      
      // Create context and access device names
      cl::Context context(platformDevices);
      ctxDevices = context.getInfo<CL_CONTEXT_DEVICES>();
      for(i=0; i<ctxDevices.size(); i++) {
         device_name = ctxDevices[i].getInfo<CL_DEVICE_NAME>();
         cout << "Device: " << device_name.c_str() << endl;
      }
   }
   catch(cl::Error e) {
      cout << e.what() << ": Error code " << e.err() << endl;
   }

   return 0;
}
