#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "transpose.cl"
#define KERNEL_FUNC "transpose"

#define MATRIX_DIM 64

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
   size_t global_size;
   cl_ulong mem_size;
   int i, j, err, check;
 
   /* Data and buffers */
   cl_uint matrix_dim;
   float data[MATRIX_DIM][MATRIX_DIM];
   cl_mem data_buffer;

   /* Initialize data */
   for(i=0; i<MATRIX_DIM; i++) {
      for(j=0; j<MATRIX_DIM; j++) {
         data[i][j] = 1.0f * i * MATRIX_DIM + j;
      }
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

   /* Create a kernel */
   kernel = clCreateKernel(program, KERNEL_FUNC, &err);
   if(err < 0) {
      perror("Couldn't create a kernel");
      exit(1);
   };

   /* Create buffer to hold matrix */
   data_buffer = clCreateBuffer(context, 
         CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
         sizeof(data), data, &err);
   if(err < 0) {
      perror("Couldn't create a buffer");
      exit(1);   
   };

   /* Determine execution parameters */
   global_size = (MATRIX_DIM/4 * (MATRIX_DIM/4 + 1))/2;
   clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, 	
         sizeof(mem_size), &mem_size, NULL);

   /* Create kernel arguments */
   matrix_dim = MATRIX_DIM/4;
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_buffer);
   err |= clSetKernelArg(kernel, 1, (size_t)mem_size, NULL);
   err |= clSetKernelArg(kernel, 2, sizeof(matrix_dim), &matrix_dim);
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
   err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, 
         NULL, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      printf("Error: %d\n", err);
      exit(1);   
   } 

   /* Read the result */
   err = clEnqueueReadBuffer(queue, data_buffer, CL_TRUE, 0, 
      sizeof(data), data, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't read the buffer");
      exit(1);   
   } 

   /* Check data */
   check = 1;
   for(i=0; i<MATRIX_DIM; i++) {
      for(j=0; j<MATRIX_DIM; j++) {
         if(data[i][j] != 1.0*j*MATRIX_DIM+i) {
            check = 0;
            break;
         }
      }
   }
   if(check)
      printf("Transpose check succeeded.\n");
   else
      printf("Transpose check failed.\n");

   /* Deallocate resources */
   clReleaseMemObject(data_buffer);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}
