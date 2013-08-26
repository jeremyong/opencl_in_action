#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "fft.cl"
#define INIT_FUNC "fft_init"
#define STAGE_FUNC "fft_stage"
#define SCALE_FUNC "fft_scale"

/* Each point contains 2 floats - 1 real, 1 imaginary */
#define NUM_POINTS 65536

/* 1 - forward FFT, -1 - inverse FFT */
#define DIRECTION 1

#include "fft_check.c"

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

   /* Host/device data structures */
   cl_device_id device;
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel init_kernel, stage_kernel, scale_kernel;
   cl_int err, i;
   size_t global_size, local_size;
   cl_ulong local_mem_size;

   /* Data and buffer */
   int direction;
   unsigned int num_points, points_per_group, stage;
   float data[NUM_POINTS*2];
   double error, check_input[NUM_POINTS][2], check_output[NUM_POINTS][2];
   cl_mem data_buffer;

   /* Initialize data */
   srand(time(NULL));
   for(i=0; i<NUM_POINTS; i++) {
      data[2*i] = rand();
      data[2*i+1] = rand();
      check_input[i][0] = data[2*i];
      check_input[i][1] = data[2*i+1];
   }

   /* Create a device and context */
   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);   
   }

   /* Build the program */
   program = build_program(context, device, PROGRAM_FILE);

   /* Create kernels for the FFT */
   init_kernel = clCreateKernel(program, INIT_FUNC, &err);
   if(err < 0) {
      printf("Couldn't create the initial kernel: %d", err);
      exit(1);
   };
   stage_kernel = clCreateKernel(program, STAGE_FUNC, &err);
   if(err < 0) {
      printf("Couldn't create the stage kernel: %d", err);
      exit(1);
   };
   scale_kernel = clCreateKernel(program, SCALE_FUNC, &err);
   if(err < 0) {
      printf("Couldn't create the scale kernel: %d", err);
      exit(1);
   };

   /* Create buffer */
   data_buffer = clCreateBuffer(context, 
         CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 
         2*NUM_POINTS*sizeof(float), data, &err);
   if(err < 0) {
      perror("Couldn't create a buffer");
      exit(1);
   };

   /* Determine maximum work-group size */
   err = clGetKernelWorkGroupInfo(init_kernel, device, 
      CL_KERNEL_WORK_GROUP_SIZE, sizeof(local_size), &local_size, NULL);
   if(err < 0) {
      perror("Couldn't find the maximum work-group size");
      exit(1);   
   };
   local_size = (int)pow(2, trunc(log2(local_size)));

   /* Determine local memory size */
   err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, 
      sizeof(local_mem_size), &local_mem_size, NULL);
   if(err < 0) {
      perror("Couldn't determine the local memory size");
      exit(1);   
   };

   /* Initialize kernel arguments */
   direction = DIRECTION;
   num_points = NUM_POINTS;
   points_per_group = local_mem_size/(2*sizeof(float));
   if(points_per_group > num_points)
      points_per_group = num_points;

   /* Set kernel arguments */
   err = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &data_buffer);
   err |= clSetKernelArg(init_kernel, 1, local_mem_size, NULL);
   err |= clSetKernelArg(init_kernel, 2, sizeof(points_per_group), &points_per_group);
   err |= clSetKernelArg(init_kernel, 3, sizeof(num_points), &num_points);
   err |= clSetKernelArg(init_kernel, 4, sizeof(direction), &direction);
   if(err < 0) {
      printf("Couldn't set a kernel argument");
      exit(1);   
   };

   /* Create a command queue */
   queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
   if(err < 0) {
      perror("Couldn't create a command queue");
      exit(1);   
   };

   /* Enqueue initial kernel */
   global_size = (num_points/points_per_group)*local_size;
   err = clEnqueueNDRangeKernel(queue, init_kernel, 1, NULL, &global_size, 
                                &local_size, 0, NULL, NULL); 
   if(err < 0) {
      perror("Couldn't enqueue the initial kernel");
      exit(1);
   }

   /* Enqueue further stages of the FFT */
   if(num_points > points_per_group) {

      err = clSetKernelArg(stage_kernel, 0, sizeof(cl_mem), &data_buffer);
      err |= clSetKernelArg(stage_kernel, 2, sizeof(points_per_group), &points_per_group);
      err |= clSetKernelArg(stage_kernel, 3, sizeof(direction), &direction);
      if(err < 0) {
         printf("Couldn't set a kernel argument");
         exit(1);   
      };
      for(stage = 2; stage <= num_points/points_per_group; stage <<= 1) {
         clSetKernelArg(stage_kernel, 1, sizeof(stage), &stage);
         err = clEnqueueNDRangeKernel(queue, stage_kernel, 1, NULL, &global_size, 
                                      &local_size, 0, NULL, NULL); 
         if(err < 0) {
            perror("Couldn't enqueue the stage kernel");
            exit(1);
         }
      }
   }

   /* Scale values if performing the inverse FFT */
   if(direction < 0) {
      err = clSetKernelArg(scale_kernel, 0, sizeof(cl_mem), &data_buffer);
      err |= clSetKernelArg(scale_kernel, 1, sizeof(points_per_group), &points_per_group);
      err |= clSetKernelArg(scale_kernel, 2, sizeof(num_points), &num_points);
      if(err < 0) {
         printf("Couldn't set a kernel argument");
         exit(1);   
      };
      err = clEnqueueNDRangeKernel(queue, scale_kernel, 1, NULL, &global_size, 
                                   &local_size, 0, NULL, NULL); 
      if(err < 0) {
         perror("Couldn't enqueue the initial kernel");
         exit(1);
      }
   }

   /* Read the results */
   err = clEnqueueReadBuffer(queue, data_buffer, CL_TRUE, 0, 
         2*NUM_POINTS*sizeof(float), data, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't read the buffer");
      exit(1);   
   }

   /* Compute accurate values */
   if(direction > 0)
      fft(NUM_POINTS, check_input, check_output);
   else
      ifft(NUM_POINTS, check_output, check_input);

   /* Determine error */
   error = 0.0;
   for(i=0; i<NUM_POINTS; i++) {
      error += fabs(check_output[i][0] - data[2*i])/fmax(fabs(check_output[i][0]), 0.0001);
      error += fabs(check_output[i][1] - data[2*i+1])/fmax(fabs(check_output[i][1]), 0.0001);
   }
   error = error/(NUM_POINTS*2);

   /* Display check results */
   printf("%u-point ", num_points);
   if(direction > 0) 
      printf("FFT ");
   else
      printf("IFFT ");
   printf("completed with %lf average relative error.\n", error);

   /* Deallocate resources */
   clReleaseMemObject(data_buffer);
   clReleaseKernel(init_kernel);
   clReleaseKernel(stage_kernel);
   clReleaseKernel(scale_kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}
