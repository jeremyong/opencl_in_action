#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "steep_desc.cl"
#define KERNEL_FUNC "steep_desc"

#define MM_FILE "../bcsstk05.mtx"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mmio.h"

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

/* Rearrange data to be sorted by row instead of by column */
void sort(int num, int *rows, int *cols, float *values) {

   int i, j, int_swap, index = 0;
   float float_swap;

   for(i=0; i<num; i++) {
      for(j=index; j<num; j++) {
         if(rows[j] == i) {
            if(j == index) {
               index++;
            }
           
            /* Swap row/column/values as necessary */
            else if(j > index) {
               int_swap = rows[index];
               rows[index] = rows[j];
               rows[j] = int_swap;

               int_swap = cols[index];
               cols[index] = cols[j];
               cols[j] = int_swap;

               float_swap = values[index];
               values[index] = values[j];
               values[j] = float_swap;
               index++;
            }
         }
      }
   }
}

int main() {

   /* Host/device data structures */
   cl_platform_id platform;
   cl_device_id device;
   cl_context context;
   cl_command_queue queue;
   cl_int err, i;

   /* Program/kernel data structures */
   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   cl_kernel kernel;
   size_t global_size, local_size;

   /* Data and buffers */
   int num_rows, num_cols, num_values;
   int *rows, *cols;
   float *values, *b_vec;
   float result[2];
   double value_double;
   cl_mem rows_buffer, cols_buffer, values_buffer, 
         b_buffer, result_buffer;

   /* Read sparse file */
   FILE *mm_handle;
   MM_typecode code;

   /* Read matrix file */   
   if ((mm_handle = fopen(MM_FILE, "r")) == NULL) {
      perror("Couldn't open the MatrixMarket file");
      exit(1);
   }
   mm_read_banner(mm_handle, &code);
   mm_read_mtx_crd_size(mm_handle, &num_rows, &num_cols, &num_values);

   /* Check for symmetry and allocate memory */
   if(mm_is_symmetric(code) || mm_is_skew(code) || mm_is_hermitian(code)) {
      num_values += num_values - num_rows;
   }
   rows = (int*) malloc(num_values * sizeof(int));
   cols = (int*) malloc(num_values * sizeof(int));
   values = (float*) malloc(num_values * sizeof(float));
   b_vec = (float*) malloc(num_rows * sizeof(float));

   /* Read matrix data and close file */
   i=0;
   while(i<num_values) {
      fscanf(mm_handle, "%d %d %lg\n", &rows[i], &cols[i], &value_double);
      values[i] = (float)value_double;
      cols[i]--;
      rows[i]--;
      if((rows[i] != cols[i]) && (mm_is_symmetric(code) || mm_is_skew(code) || mm_is_hermitian(code))) {
         i++;
         rows[i] = cols[i-1];
         cols[i] = rows[i-1];
         values[i] = values[i-1];
      }
      i++;
   }
   sort(num_values, rows, cols, values);
   fclose(mm_handle);

   /* Initialize the b vector */
   srand(time(0));
   for(i=0; i<num_rows; i++) {
      b_vec[i] = (float)rand()/RAND_MAX;
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

   /* Create buffers to hold the text characters and count */
   rows_buffer = clCreateBuffer(context, 
         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
         num_values*sizeof(int), rows, &err);
   if(err < 0) {
      perror("Couldn't create a buffer");
      exit(1);   
   };
   cols_buffer = clCreateBuffer(context,
         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
         num_values*sizeof(int), cols, NULL);
   values_buffer = clCreateBuffer(context,
         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
         num_values*sizeof(float), values, NULL);
   b_buffer = clCreateBuffer(context,
         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
         num_values*sizeof(float), b_vec, NULL);
   result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
         2*sizeof(float), NULL, NULL);

   /* Create kernel argument */
   err = clSetKernelArg(kernel, 0, sizeof(num_rows), &num_rows);
   err |= clSetKernelArg(kernel, 1, sizeof(num_values), &num_values);
   err |= clSetKernelArg(kernel, 2, num_rows * sizeof(float), NULL);
   err |= clSetKernelArg(kernel, 3, num_rows * sizeof(float), NULL);
   err |= clSetKernelArg(kernel, 4, num_rows * sizeof(float), NULL);
   err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &rows_buffer);
   err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &cols_buffer);
   err |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &values_buffer);
   err |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &b_buffer);
   err |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &result_buffer);
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
   global_size = num_rows;
   local_size = num_rows;
   err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      printf("Error: %d\n", err);
      exit(1);
   }

   /* Read the results */
   err = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0, 
      2*sizeof(float), result, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't read the buffer");
      exit(1);   
   }

   /* Print the result */
   printf("After %d iterations, the residual length is %e.\n", 
         (int)result[0], result[1]);

   /* Deallocate resources */
   free(b_vec);
   free(rows);
   free(cols);
   free(values);
   clReleaseMemObject(b_buffer);
   clReleaseMemObject(rows_buffer);
   clReleaseMemObject(cols_buffer);
   clReleaseMemObject(values_buffer);
   clReleaseMemObject(result_buffer);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}
