#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "test.cl"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

int main() {

   /* Host/device data structures */
   cl_platform_id platform;
   cl_device_id device;
   cl_context context;
   cl_program program;
   cl_int err;

   /* Program/kernel data structures */
   FILE *program_handle;   
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   cl_kernel *kernels, found_kernel;
   char kernel_name[20];
   cl_uint i, num_kernels;

   /* Access the first installed platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't find any platforms");
      exit(1);
   }

   /* Access the first available device */
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
   }
   if(err < 0) {
      perror("Couldn't find any devices");
      exit(1);
   }

   /* Create a context */
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);   
   }

   /* Read program file and place content into buffer */
   program_handle = fopen(PROGRAM_FILE, "r");
   if(program_handle == NULL) {
      perror("Couldn't find the program file");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size+1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   /* Create program from file */
   program = clCreateProgramWithSource(context, 1, 
      (const char**)&program_buffer, &program_size, &err);
   if(err < 0) {
      perror("Couldn't create the program");
      exit(1);   
   }
   free(program_buffer);


   /* Build program */
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if(err < 0) {

      /* Find size of log and print to std output */
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
            0, NULL, &log_size);
      program_log = (char*) malloc(log_size+1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
            log_size+1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   /* Find out how many kernels are in the source file */
   err = clCreateKernelsInProgram(program, 0, NULL, &num_kernels);	
   if(err < 0) {
      perror("Couldn't find any kernels");
      exit(1);  
   }

   /* Create a kernel for each function */
   kernels = (cl_kernel*) malloc(num_kernels * sizeof(cl_kernel));
   clCreateKernelsInProgram(program, num_kernels, kernels, NULL);	

   /* Search for the named kernel */
   for(i=0; i<num_kernels; i++) {					
      clGetKernelInfo(kernels[i], CL_KERNEL_FUNCTION_NAME, 		
            sizeof(kernel_name), kernel_name, NULL);				
      if(strcmp(kernel_name, "mult") == 0) {
         found_kernel = kernels[i];
         printf("Found mult kernel at index %u.\n", i);
         break;
      }									
   }									

   for(i=0; i<num_kernels; i++)					
      clReleaseKernel(kernels[i]);
   free(kernels);
   clReleaseProgram(program);
   clReleaseContext(context);
}
