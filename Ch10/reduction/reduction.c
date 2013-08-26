#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "reduction.cl"

#define ARRAY_SIZE 1048576
#define NUM_KERNELS 2

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

   /* OpenCL structures */
   cl_device_id device;
   cl_context context;
   cl_program program;
   cl_kernel kernel[NUM_KERNELS];
   cl_command_queue queue;
   cl_event prof_event;
   cl_int i, j, err;
   size_t local_size, global_size;
   char kernel_names[NUM_KERNELS][20] = 
         {"reduction_scalar", "reduction_vector"};

   /* Data and buffers */
   float data[ARRAY_SIZE];
   float sum, actual_sum, *scalar_sum, *vector_sum;
   cl_mem data_buffer, scalar_sum_buffer, vector_sum_buffer;
   cl_int num_groups;
   cl_ulong time_start, time_end, total_time;

   /* Initialize data */
   for(i=0; i<ARRAY_SIZE; i++) {
      data[i] = 1.0f*i;
   }

   /* Create device and determine local size */
   device = create_device();
   err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, 	
         sizeof(local_size), &local_size, NULL);	
   if(err < 0) {
      perror("Couldn't obtain device information");
      exit(1);   
   }

   /* Allocate and initialize output arrays */
   num_groups = ARRAY_SIZE/local_size;
   scalar_sum = (float*) malloc(num_groups * sizeof(float));
   vector_sum = (float*) malloc(num_groups/4 * sizeof(float));
   for(i=0; i<num_groups; i++) {
      scalar_sum[i] = 0.0f;
   }
   for(i=0; i<num_groups/4; i++) {
      vector_sum[i] = 0.0f;
   }

   /* Create a context */
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);   
   }

   /* Build program */
   program = build_program(context, device, PROGRAM_FILE);

   /* Create data buffer */
   data_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
         CL_MEM_COPY_HOST_PTR, ARRAY_SIZE * sizeof(float), data, &err);
   scalar_sum_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
         CL_MEM_COPY_HOST_PTR, num_groups * sizeof(float), scalar_sum, &err);
   vector_sum_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
         CL_MEM_COPY_HOST_PTR, num_groups * sizeof(float), vector_sum, &err);
   if(err < 0) {
      perror("Couldn't create a buffer");
      exit(1);   
   };

   /* Create a command queue */
   queue = clCreateCommandQueue(context, device, 
         CL_QUEUE_PROFILING_ENABLE, &err);
   if(err < 0) {
      perror("Couldn't create a command queue");
      exit(1);   
   };

   for(i=0; i<NUM_KERNELS; i++) {

      /* Create a kernel */
      kernel[i] = clCreateKernel(program, kernel_names[i], &err);
      if(err < 0) {
         perror("Couldn't create a kernel");
         exit(1);
      };

      /* Create kernel arguments */
      err = clSetKernelArg(kernel[i], 0, sizeof(cl_mem), &data_buffer);
      if(i == 0) {
         global_size = ARRAY_SIZE;
         err |= clSetKernelArg(kernel[i], 1, local_size * sizeof(float), NULL);
         err |= clSetKernelArg(kernel[i], 2, sizeof(cl_mem), &scalar_sum_buffer);
      }
      else {
         global_size = ARRAY_SIZE/4;
         err |= clSetKernelArg(kernel[i], 1, local_size * 4 * sizeof(float), NULL);
         err |= clSetKernelArg(kernel[i], 2, sizeof(cl_mem), &vector_sum_buffer);
      }
      if(err < 0) {
         perror("Couldn't create a kernel argument");
         exit(1);   
      }

      /* Enqueue kernel */
      err = clEnqueueNDRangeKernel(queue, kernel[i], 1, NULL, &global_size, 
            &local_size, 0, NULL, &prof_event); 
      if(err < 0) {
         perror("Couldn't enqueue the kernel");
         exit(1);   
      }

      /* Finish processing the queue and get profiling information */
      clFinish(queue);
      clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START,
            sizeof(time_start), &time_start, NULL);
      clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END,
            sizeof(time_end), &time_end, NULL);
      total_time = time_end - time_start;

      /* Read the result */
      if(i == 0) {
         err = clEnqueueReadBuffer(queue, scalar_sum_buffer, CL_TRUE, 0, 
            num_groups * sizeof(float), scalar_sum, 0, NULL, NULL);
         if(err < 0) {
            perror("Couldn't read the buffer");
            exit(1);   
         }
         sum = 0.0f;
         for(j=0; j<num_groups; j++) {
            sum += scalar_sum[j];
         }
      }
      else {
         err = clEnqueueReadBuffer(queue, vector_sum_buffer, CL_TRUE, 0, 
            num_groups/4 * sizeof(float), vector_sum, 0, NULL, NULL);
         if(err < 0) {
            perror("Couldn't read the buffer");
            exit(1);   
         }
         sum = 0.0f;
         for(j=0; j<num_groups/4; j++) {
            sum += vector_sum[j];
         }
      }

      /* Check result */
      printf("%s: ", kernel_names[i]);
      actual_sum = 1.0f * ARRAY_SIZE/2*(ARRAY_SIZE-1);
      if(fabs(sum - actual_sum) > 0.01*fabs(sum))
         printf("Check failed.\n");
      else
         printf("Check passed.\n");
      printf("Total time = %lu\n\n", total_time);

      /* Deallocate event */
      clReleaseEvent(prof_event);
   }

   /* Deallocate resources */
   free(scalar_sum);
   free(vector_sum);
   for(i=0; i<NUM_KERNELS; i++) {
      clReleaseKernel(kernel[i]);
   }
   clReleaseMemObject(scalar_sum_buffer);
   clReleaseMemObject(vector_sum_buffer);
   clReleaseMemObject(data_buffer);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}
