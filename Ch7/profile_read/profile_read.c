#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "profile_read.cl"
#define KERNEL_FUNC "profile_read"

#define NUM_BYTES 131072
#define NUM_ITERATIONS 2000
//#define PROFILE_READ 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device() {

   cl_platform_id platform;
   cl_device_id dev;
   int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't identify a platform");
      exit(1);
   } 

   /* Access a device */
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   if(err < 0) {
      perror("Couldn't access any devices");
      exit(1);   
   }

   return dev;
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   int err;

   /* Read program file and place content into buffer */
   program_handle = fopen(filename, "r");
   if(program_handle == NULL) {
      perror("Couldn't find the program file");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   /* Create program from file */
   program = clCreateProgramWithSource(ctx, 1, 
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
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   return program;
}

int main() {

   /* OpenCL data structures */
   cl_device_id device;
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel kernel;
   cl_int i, err, num_vectors;

   /* Data and events */
   char data[NUM_BYTES];
   cl_mem data_buffer;
   cl_event prof_event;
   cl_ulong time_start, time_end, total_time;
   void* mapped_memory;

   /* Create a device and context */
   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);   
   }     

   /* Build the program and create a kernel */
   program = build_program(context, device, PROGRAM_FILE);
   kernel = clCreateKernel(program, KERNEL_FUNC, &err);
   if(err < 0) {
      perror("Couldn't create a kernel");
      exit(1);   
   };

   /* Create a buffer to hold data */
   data_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
         sizeof(data), NULL, &err);
   if(err < 0) {
      perror("Couldn't create a buffer");
      exit(1);   
   };         

   /* Create kernel argument */
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_buffer);
   if(err < 0) {
      perror("Couldn't set a kernel argument");
      exit(1);   
   };

   /* Tell kernel number of char16 vectors */
   num_vectors = NUM_BYTES/16;
   clSetKernelArg(kernel, 1, sizeof(num_vectors), &num_vectors);

   /* Create a command queue */
   queue = clCreateCommandQueue(context, device, 
         CL_QUEUE_PROFILING_ENABLE, &err);
   if(err < 0) {
      perror("Couldn't create a command queue");
      exit(1);   
   };

   total_time = 0.0f;
   for(i=0; i<NUM_ITERATIONS; i++) {

      /* Enqueue kernel */
      err = clEnqueueTask(queue, kernel, 0, NULL, NULL);
      if(err < 0) {
         perror("Couldn't enqueue the kernel");
         exit(1);   
      }

#ifdef PROFILE_READ

      /* Read the buffer */
      err = clEnqueueReadBuffer(queue, data_buffer, CL_TRUE, 0, 
            sizeof(data), data, 0, NULL, &prof_event);
      if(err < 0) {
         perror("Couldn't read the buffer");
         exit(1);
      }

#else

      /* Create memory map */
      mapped_memory = clEnqueueMapBuffer(queue, data_buffer, CL_TRUE,
            CL_MAP_READ, 0, sizeof(data), 0, NULL, &prof_event, &err);
      if(err < 0) {
         perror("Couldn't map the buffer to host memory");
         exit(1);   
      }

#endif

      /* Get profiling information */
      clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START,
            sizeof(time_start), &time_start, NULL);
      clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END,
            sizeof(time_end), &time_end, NULL);
      total_time += time_end - time_start;

#ifndef PROFILE_READ

      /* Unmap the buffer */
      err = clEnqueueUnmapMemObject(queue, data_buffer, mapped_memory,
            0, NULL, NULL);
      if(err < 0) {
         perror("Couldn't unmap the buffer");
         exit(1);   
      }

#endif
   }

#ifdef PROFILE_READ
   printf("Average read time: %lu\n", total_time/NUM_ITERATIONS);
#else
   printf("Average map time: %lu\n", total_time/NUM_ITERATIONS);
#endif

   /* Deallocate resources */
   clReleaseEvent(prof_event);
   clReleaseMemObject(data_buffer);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}
