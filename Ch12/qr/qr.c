#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "qr.cl"
#define KERNEL_FUNC "qr"

#define MATRIX_DIM 32

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
   cl_kernel kernel;
   size_t global_size, local_size;
   cl_int err, i, j, k, check;

   /* Data and buffers */
   float a_mat[MATRIX_DIM][MATRIX_DIM], q_mat[MATRIX_DIM][MATRIX_DIM], 
         r_mat[MATRIX_DIM][MATRIX_DIM], check_mat[MATRIX_DIM][MATRIX_DIM];
   cl_mem a_buffer, q_buffer, p_buffer, prod_buffer;

   /* Initialize A matrix */
   srand((unsigned int)time(0));
   for(i=0; i<MATRIX_DIM; i++) {
      for(j=0; j<MATRIX_DIM; j++) {
         a_mat[i][j] = (float)rand()/RAND_MAX;
         check_mat[i][j] = 0.0f;
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

   /* Create buffer */
   a_buffer = clCreateBuffer(context, 
         CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
         sizeof(a_mat), a_mat, &err);
   if(err < 0) {
      perror("Couldn't create a buffer");
      exit(1);   
   };
   q_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
         sizeof(q_mat), NULL, NULL);
   p_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
         sizeof(q_mat), NULL, NULL);
   prod_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
         sizeof(q_mat), NULL, NULL);

   /* Create kernel arguments */
   err = clSetKernelArg(kernel, 0, MATRIX_DIM * sizeof(float), NULL);
   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &a_buffer);
   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &q_buffer);
   err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &p_buffer);
   err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &prod_buffer);
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
   global_size = MATRIX_DIM;
   local_size = MATRIX_DIM;
   err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      exit(1);
   }

   /* Read the results */
   err = clEnqueueReadBuffer(queue, q_buffer, CL_TRUE, 0, 
      sizeof(q_mat), q_mat, 0, NULL, NULL);
   err |= clEnqueueReadBuffer(queue, a_buffer, CL_TRUE, 0, 
      sizeof(r_mat), r_mat, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't read the buffers");
      exit(1);   
   }

   /* Compute product of Q and R */
   for(i=0; i<MATRIX_DIM; i++) {
      for(j=0; j<MATRIX_DIM; j++) {
         for(k=0; k<MATRIX_DIM; k++) {
            check_mat[i][j] += q_mat[i][k] * r_mat[k][j];
         }
      }
   }

   /* Check data */
   check = 1;
   for(i=0; i<MATRIX_DIM; i++) {
      for(j=0; j<MATRIX_DIM; j++) {
         if(fabs(a_mat[i][j] - check_mat[i][j]) > 0.01f) {
            check = 0;
            break;
         }
      }
   }
   if(check)
      printf("QR decomposition check succeeded.\n");
   else
      printf("QR decomposition check failed.\n"); 

   /* Deallocate resources */
   clReleaseMemObject(a_buffer);
   clReleaseMemObject(q_buffer);
   clReleaseMemObject(p_buffer);
   clReleaseMemObject(prod_buffer);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}
