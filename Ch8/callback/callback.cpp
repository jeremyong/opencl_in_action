#define __CL_ENABLE_EXCEPTIONS
#define __NO_STD_VECTOR
#define PROGRAM_FILE "blank.cl"
#define KERNEL_FUNC "blank"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

void CL_CALLBACK checkData(cl_event event, cl_int status, void* data) {

   int i;
   cl_bool check;
   int *buffer_data;

   buffer_data = (int*)data;
   check = CL_TRUE;
   for(i=0; i<100; i++) {
      if(buffer_data[i] != 2*i) {
         check = CL_FALSE;
         break;
      }  
   }
   if(check)
      std::cout << "The data is accurate." << std::endl;
   else
      std::cout << "The data is not accurate." << std::endl;
}

int main(void) {
   
   cl::vector<cl::Platform> platforms;
   cl::vector<cl::Device> devices;
   cl::Event callbackEvent;
   int i, data[100];

   try {
      // Initialize data
      for(i=0; i<100; i++) {
         data[i] = i;
      }

      // Place the GPU devices of the first platform into a context
      cl::Platform::get(&platforms);
      platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
      cl::Context context(devices);
      
      // Create kernel
      std::ifstream programFile(PROGRAM_FILE);
      std::string programString(std::istreambuf_iterator<char>(programFile),
            (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1, std::make_pair(programString.c_str(),
            programString.length()+1));
      cl::Program program(context, source);
      program.build(devices);
      cl::Kernel kernel(program, KERNEL_FUNC);

      // Create buffer and make it a kernel argument
      cl::Buffer buffer(context, 
         CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
         sizeof(data), data);
      kernel.setArg(0, buffer);

      // Create queue and enqueue kernel-execution command
      cl::CommandQueue queue(context, devices[0]);
      queue.enqueueTask(kernel);

      // Read output buffer from kernel
      queue.enqueueReadBuffer(buffer, CL_FALSE, 0, sizeof(data), 
            data, NULL, &callbackEvent);

      // Set callback function
      callbackEvent.setCallback(CL_COMPLETE, &checkData, (void*)data);
   }
   catch(cl::Error e) {
      std::cout << e.what() << ": Error code " << e.err() << std::endl;   
   }

   return 0;
}
