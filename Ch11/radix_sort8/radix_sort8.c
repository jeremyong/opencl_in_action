#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "radix_sort8.cl"
#define KERNEL_FUNC "radix_sort8"

#define NUM_SHORTS 8

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
   cl_int i, j, check, temp, err;

   /* Program/kernel data structures */
   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   cl_kernel kernel;     

   /* Data and buffers */
   unsigned short data[NUM_SHORTS];
   cl_mem data_buffer;
   
   /* Initialize data */
   srand(time(NULL));
   for(i=0; i<NUM_SHORTS; i++) {
      data[i] = (unsigned short)i;
   }
   for(i=0; i<NUM_SHORTS-1; i++) {
      j = i + (rand() % (NUM_SHORTS-i));
      temp = data[i]; data[i] = data[j]; data[j] = temp;
   }

   /* Print input */
   printf("Input: \n");
   for(i=0; i<NUM_SHORTS; i++) {
      printf("data[%d]: %hu\n", i, data[i]);
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
      perror("Couldn't create a kernel");
      exit(1);
   };

   /* Create buffer to hold sorted data */
   data_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
         CL_MEM_COPY_HOST_PTR, sizeof(data), data, &err);
   if(err < 0) {
      perror("Couldn't create a buffer");
      exit(1);   
   };

   /* Create kernel argument */
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
   err = clEnqueueTask(queue, kernel, 0, NULL, NULL); 
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      exit(1);   
   }

   /* Read and print the result */
   err = clEnqueueReadBuffer(queue, data_buffer, CL_TRUE, 0, 
      sizeof(data), &data, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't read the buffer");
      exit(1);   
   }

   /* Print output */
   printf("Output: \n");
   for(i=0; i<NUM_SHORTS; i++) {
      printf("data[%d]: %hu\n", i, data[i]);
   }

   /* Check the output and display test result */
   check = 1;
   for(i=0; i<NUM_SHORTS; i++) {
      if(data[i] != i) {
         check = 0;
         break;
      }
   }
   if(check)
      printf("The radix sort succeeded.\n");
   else
      printf("The radix sort failed.\n");

   /* Deallocate resources */
   clReleaseMemObject(data_buffer);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}
