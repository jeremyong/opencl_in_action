#define _CRT_SECURE_NO_WARNINGS
#define M_PI 3.14159265358979323846

#define PROGRAM_FILE "polar_rect.cl"
#define KERNEL_FUNC "polar_rect"

#include <math.h>
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

   /* Host/device data structures */
   cl_device_id device;
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel kernel;
   cl_int i, err;

   /* Data and buffers */
   float r_coords[4] = {2, 1, 3, 4};
   float angles[4] = {3*M_PI/8, 3*M_PI/4, 4*M_PI/3, 11*M_PI/6};
   float x_coords[4], y_coords[4];
   cl_mem r_coords_buffer, angles_buffer,
         x_coords_buffer, y_coords_buffer;

   /* Create a device and context */
   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);   
   }

   /* Create a kernel */
   program = build_program(context, device, PROGRAM_FILE);
   kernel = clCreateKernel(program, KERNEL_FUNC, &err);
   if(err < 0) {
      perror("Couldn't create a kernel");
      exit(1);   
   };

   /* Create a write-only buffer to hold the output data */
   r_coords_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |  CL_MEM_COPY_HOST_PTR,
         sizeof(r_coords), r_coords, &err);
   if(err < 0) {
      perror("Couldn't create a buffer");
      exit(1);   
   };
   angles_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |  CL_MEM_COPY_HOST_PTR,
         sizeof(angles), angles, &err);   
   x_coords_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 
         sizeof(x_coords), NULL, &err);
   y_coords_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 
         sizeof(y_coords), NULL, &err);         
         
   /* Create kernel argument */
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &r_coords_buffer);
   if(err < 0) {
      perror("Couldn't set a kernel argument");
      exit(1);   
   };
   clSetKernelArg(kernel, 1, sizeof(cl_mem), &angles_buffer);
   clSetKernelArg(kernel, 2, sizeof(cl_mem), &x_coords_buffer);
   clSetKernelArg(kernel, 3, sizeof(cl_mem), &y_coords_buffer);
   
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
   err = clEnqueueReadBuffer(queue, x_coords_buffer, CL_TRUE, 0, 
      sizeof(x_coords), &x_coords, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't read the buffer");
      exit(1);   
   }
   clEnqueueReadBuffer(queue, y_coords_buffer, CL_TRUE, 0, 
      sizeof(y_coords), &y_coords, 0, NULL, NULL);   

   /* Display the results */
   for(i=0; i<4; i++) {
      printf("(%6.3f, %6.3f)\n", x_coords[i], y_coords[i]);
   }   
      
   /* Deallocate resources */
   clReleaseMemObject(r_coords_buffer);
   clReleaseMemObject(angles_buffer);  
   clReleaseMemObject(x_coords_buffer);
   clReleaseMemObject(y_coords_buffer);   
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}
