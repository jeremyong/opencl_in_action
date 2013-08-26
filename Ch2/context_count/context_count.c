#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
   cl_int err;
   cl_uint ref_count;

   /* Access the first installed platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't find any platforms");
      exit(1);
   }

   /* Access the first available device */
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
   }
   if(err < 0) {
      perror("Couldn't find any devices");
      exit(1);
   }

   /* Create the context */
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);   
   }

   /* Determine the reference count */
   err = clGetContextInfo(context, CL_CONTEXT_REFERENCE_COUNT, 	
         sizeof(ref_count), &ref_count, NULL);			
   if(err < 0) {		
      perror("Couldn't read the reference count.");
      exit(1);
   }
   printf("Initial reference count: %u\n", ref_count);

   /* Update and display the reference count */
   clRetainContext(context);						
   clGetContextInfo(context, CL_CONTEXT_REFERENCE_COUNT, 		
         sizeof(ref_count), &ref_count, NULL);			
   printf("Reference count: %u\n", ref_count);			
   
   clReleaseContext(context);						
   clGetContextInfo(context, CL_CONTEXT_REFERENCE_COUNT, 		
         sizeof(ref_count), &ref_count, NULL);			
   printf("Reference count: %u\n", ref_count);			

   clReleaseContext(context);						
   return 0;
}
