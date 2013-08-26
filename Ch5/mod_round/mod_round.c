#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "mod_round.cl"
#define KERNEL_FUNC "mod_round"

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
   cl_int err;

   /* Data and buffers */
   float mod_input[2] = {317.0f, 23.0f};
   float mod_output[2];   
   float round_input[4] = {-6.5f, -3.5f, 3.5f, 6.5f};
   float round_output[20];
   cl_mem mod_input_buffer, mod_output_buffer,
         round_input_buffer, round_output_buffer;

   /* Create a context */
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

   /* Create buffers to hold input/output data */
   mod_input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
         sizeof(mod_input), mod_input, &err);
   if(err < 0) {
      perror("Couldn't create a buffer");
      exit(1);   
   };
   mod_output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
         sizeof(mod_output), NULL, NULL); 
   round_input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
         sizeof(round_input), round_input, NULL);         
   round_output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
         sizeof(round_output), NULL, NULL);

   /* Create kernel argument */
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mod_input_buffer);
   if(err < 0) {
      perror("Couldn't set a kernel argument");
      exit(1);   
   };
   clSetKernelArg(kernel, 1, sizeof(cl_mem), &mod_output_buffer);
   clSetKernelArg(kernel, 2, sizeof(cl_mem), &round_input_buffer);
   clSetKernelArg(kernel, 3, sizeof(cl_mem), &round_output_buffer);   

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

   /* Read the results */
   err = clEnqueueReadBuffer(queue, mod_output_buffer, CL_TRUE, 0, 
      sizeof(mod_output), &mod_output, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't read the buffer");
      exit(1);   
   }
   clEnqueueReadBuffer(queue, round_output_buffer, CL_TRUE, 0, 
      sizeof(round_output), &round_output, 0, NULL, NULL);
      
   /* Display data */
   printf("fmod(%.1f, %.1f)      = %.1f\n", mod_input[0], mod_input[1], mod_output[0]);
   printf("remainder(%.1f, %.1f) = %.1f\n\n", mod_input[0], mod_input[1], mod_output[1]);
   
   printf("Rounding input: %.1f %.1f %.1f %.1f\n", 
         round_input[0], round_input[1], round_input[2], round_input[3]);
   printf("rint:  %.1f, %.1f, %.1f, %.1f\n", 
         round_output[0], round_output[1], round_output[2], round_output[3]);
   printf("round: %.1f, %.1f, %.1f, %.1f\n", 
         round_output[4], round_output[5], round_output[6], round_output[7]);
   printf("ceil:  %.1f, %.1f, %.1f, %.1f\n", 
         round_output[8], round_output[9], round_output[10], round_output[11]);
   printf("floor: %.1f, %.1f, %.1f, %.1f\n", 
         round_output[12], round_output[13], round_output[14], round_output[15]);
   printf("trunc: %.1f, %.1f, %.1f, %.1f\n", 
         round_output[16], round_output[17], round_output[18], round_output[19]);         


   /* Deallocate resources */
   clReleaseMemObject(mod_input_buffer);
   clReleaseMemObject(mod_output_buffer);
   clReleaseMemObject(round_input_buffer);
   clReleaseMemObject(round_output_buffer);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}
