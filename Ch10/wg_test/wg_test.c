#define _CRT_SECURE_NO_WARNINGS
#define DEFAULT_PROGRAM "blank.cl"
#define DEFAULT_KERNEL "blank"

#include <stdio.h>
#include <stdlib.h>

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

int main(int argc, char **argv) {

   /* Host/device structures */
   cl_device_id device;
   cl_context context;
   cl_program program;
   cl_kernel kernel;
   char *program_name, *kernel_name;
   cl_command_queue queue;
   cl_int err;

   /* Device/Kernel data */
   char device_name[48];
   size_t wg_size, wg_multiple;
   cl_ulong local_mem, private_usage, local_usage;

   /* Read program name and kernel name from input arguments */
   switch(argc) {
      case 1:
         program_name = DEFAULT_PROGRAM;
         kernel_name = DEFAULT_KERNEL;
      break;
      case 3:
         program_name = argv[1];
         kernel_name = argv[2];
      break;
      default:
         printf("Usage: wg_test <program_file> <kernel_name>\n");
         exit(1);
      break;
   }

   /* Access device properties */
   device = create_device();
   err = clGetDeviceInfo(device, CL_DEVICE_NAME, 		
         sizeof(device_name), device_name, NULL);   
   err |= clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, 		
         sizeof(local_mem), &local_mem, NULL);	          
   if(err < 0) {
      perror("Couldn't obtain device information");
      exit(1);   
   }   
   
   /* Create a context */
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);   
   }

   /* Build program */
   program = build_program(context, device, program_name);

   /* Create a command queue */
   queue = clCreateCommandQueue(context, device, 0, &err);
   if(err < 0) {
      perror("Couldn't create a command queue");
      exit(1);   
   };

   /* Create a kernel */
   kernel = clCreateKernel(program, kernel_name, &err);
   if(err < 0) {
      perror("Couldn't create a kernel");
      exit(1);
   };

   /* Access kernel/work-group properties */
   err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
      sizeof(wg_size), &wg_size, NULL);
   err |= clGetKernelWorkGroupInfo(kernel, device, 
      CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
      sizeof(wg_multiple), &wg_multiple, NULL);
   err |= clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_LOCAL_MEM_SIZE,
      sizeof(local_usage), &local_usage, NULL);
   err |= clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_PRIVATE_MEM_SIZE,
      sizeof(private_usage), &private_usage, NULL);
   if(err < 0) {
      perror("Couldn't obtain kernel work-group size information");
      exit(1);
   };

   /* Display results */
   printf("For the %s kernel running on the %s device, the maximum work-group size is %zu and the work-group multiple is %zu.\n\n", 
         kernel_name, device_name, wg_size, wg_multiple);
   printf("The kernel uses %zu bytes of local memory out of a maximum of %zu bytes. It uses %zu bytes of private memory.\n", 
         local_usage, local_mem, private_usage);

   /* Deallocate resources */
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}
