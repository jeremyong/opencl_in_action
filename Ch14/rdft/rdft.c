#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "rdft.cl"
#define KERNEL_FUNC "rdft"

#define NUM_POINTS 256

#include "fft_check.c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
   cl_command_queue queue;
   cl_int err, i, check;

   /* Program/kernel data structures */
   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   cl_kernel kernel;
   size_t global_size, local_size;

   /* Data and buffer */
   float input[NUM_POINTS], output[NUM_POINTS];
   double check_input[NUM_POINTS][2], check_output[NUM_POINTS][2];
   cl_mem data_buffer;

   /* Initialize data with a rectangle function */   
   for(int i=0; i<NUM_POINTS/4; i++) {
      input[i] = 1.0f;
      check_input[i][0] = 1.0;
      check_input[i][1] = 0.0;
   }
   for(int i=NUM_POINTS/4; i<NUM_POINTS; i++) {
      input[i] = 0.0f;
      check_input[i][0] = 0.0;
      check_input[i][1] = 0.0;
   }

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't identify a platform");
      exit(1);
   } 

   /* Access a device */
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
   if(err < 0) {
      perror("Couldn't access any devices");
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
   program_buffer = (char*)calloc(program_size+1, sizeof(char));
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
      program_log = (char*) calloc(log_size+1, sizeof(char));
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
            log_size+1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   /* Create a kernel */
   kernel = clCreateKernel(program, KERNEL_FUNC, &err);
   if(err < 0) {
      printf("Couldn't create a kernel: %d", err);
      exit(1);
   };

   /* Create buffers */
   data_buffer = clCreateBuffer(context, 
         CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
         NUM_POINTS*sizeof(float), input, &err);
   if(err < 0) {
      perror("Couldn't create a buffer");
      exit(1);
   };

   /* Create kernel arguments */
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_buffer);
   if(err < 0) {
      printf("Couldn't set a kernel argument");
      exit(1);   
   };

   /* Create a command queue */
   queue = clCreateCommandQueue(context, device, 0, &err);
   if(err < 0) {
      perror("Couldn't create a command queue");
      exit(1);   
   };

   /* Enqueue kernel */
   global_size = (NUM_POINTS/2)+1;
   local_size = (NUM_POINTS/2)+1;
   err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      exit(1);
   }

   /* Read the results */
   err = clEnqueueReadBuffer(queue, data_buffer, CL_TRUE, 0, 
      NUM_POINTS*sizeof(float), output, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't read the buffer");
      exit(1);   
   }

   /* Check the results */
   check = 1;
   fft(NUM_POINTS, check_input, check_output);
   if((fabs(output[0] - check_output[0][0]) > 0.001) || 
         (fabs(output[1] - check_output[NUM_POINTS/2][0]) > 0.001)) {
      check = 0;
   }
   for(i=2; i<NUM_POINTS/2; i+=2) {
      if((fabs(output[i] - check_output[i/2][0]) > 0.001) || 
            (fabs(output[i+1] - check_output[i/2][1]) > 0.001)) {
         check = 0;
         break;
      } 
   }
   if(check)
      printf("Real-valued DFT check succeeded.\n");
   else
      printf("Real-valued DFT check failed.\n");

   /* Deallocate resources */
   clReleaseMemObject(data_buffer);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}
