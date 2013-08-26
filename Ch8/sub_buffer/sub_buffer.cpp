#define __CL_ENABLE_EXCEPTIONS
#define __NO_STD_VECTOR
#define PROGRAM_FILE "sub_buffer.cl"
#define KERNEL_FUNC "sub_buffer"

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
   cl::vector<cl::Kernel> allKernels;
   std::string kernelName;
   cl::Buffer mainBuffer, subBuffer;
   float data[200];

   try {
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

      // Create main buffer and make it the kernel's first argument
      cl::Buffer mainBuffer(context, 
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(data), data);
      kernel.setArg(0, mainBuffer);

      // Create sub-buffer and make it the kernel's second argument
      size_t config[2] = {70*sizeof(float), 20*sizeof(float)};
      subBuffer = mainBuffer.createSubBuffer(
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
            CL_BUFFER_CREATE_TYPE_REGION, (void*)config);
      kernel.setArg(1, subBuffer);

      // Display sizes and locations of buffers
      std::cout << "Main buffer size: " 
            << mainBuffer.getInfo<CL_MEM_SIZE>() << std::endl;
      std::cout << "Main buffer memory location: " 
            << mainBuffer.getInfo<CL_MEM_HOST_PTR>() << std::endl;
      std::cout << "Sub-buffer size: " 
            << subBuffer.getInfo<CL_MEM_SIZE>() << std::endl;
      std::cout << "Sub-buffer memory location: " 
            << subBuffer.getInfo<CL_MEM_HOST_PTR>() << std::endl;
   }
   catch(cl::Error e) {
      std::cout << e.what() << ": Error code " << e.err() << std::endl;   
   }

   return 0;
}
