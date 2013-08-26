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

int main(void) {
   
   cl::vector<cl::Platform> platforms;
   cl::vector<cl::Device> devices;
   int i, j ;

   // Data
   float dataA[100], dataB[100], results[100];
   void* mappedMemory;

   try {
      // Initialize data
      for(i=0; i<100; i++) {
         dataA[i] = i*1.0f;
         dataB[i] = i*-1.0f;
         results[i] = 0.0f;
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

      // Create buffers
      cl::Buffer bufferA(context, 
         CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
         sizeof(dataA), dataA);
      cl::Buffer bufferB(context, 
         CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
         sizeof(dataB), dataB);

      // Set kernel arguments
      kernel.setArg(0, bufferA);
      kernel.setArg(1, bufferB);

      // Create queue and enqueue kernel-execution command
      cl::CommandQueue queue(context, devices[0]);
      queue.enqueueTask(kernel);

      // Copy Buffer A to Buffer B
      queue.enqueueCopyBuffer(bufferA, bufferB, 0, 0, sizeof(dataA));

      // Map Buffer B to the host
      mappedMemory = queue.enqueueMapBuffer(bufferB, CL_TRUE, 
            CL_MAP_READ, 0, sizeof(dataB));

      // Transfer memory on the host and unmap the buffer
      memcpy(results, mappedMemory, sizeof(dataB));
      queue.enqueueUnmapMemObject(bufferB, mappedMemory);

      // Display updated buffer
      for(i=0; i<10; i++) {
         for(j=0; j<10; j++) {
            printf("%6.1f", results[j+i*10]);
         }
         printf("\n");
      }
   }
   catch(cl::Error e) {
      std::cout << e.what() << ": Error code " << e.err() << std::endl;   
   }

   return 0;
}
