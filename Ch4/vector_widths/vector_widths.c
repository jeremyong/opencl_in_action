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
   cl_uint vector_width;
   cl_int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't find any platforms");
      exit(1);
   }

   /* Access a device */
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
   }
   if(err < 0) {
      perror("Couldn't access any devices");
      exit(1);   
   }
   
   /* Obtain the device data */
   err = clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, 
         sizeof(vector_width), &vector_width, NULL);     
   if(err < 0) {
      perror("Couldn't read device properties");
      exit(1);
   }
   printf("Preferred vector width in chars: %u\n", vector_width);
   clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, 
         sizeof(vector_width), &vector_width, NULL);      
   printf("Preferred vector width in shorts: %u\n", vector_width);
   clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, 
         sizeof(vector_width), &vector_width, NULL);      
   printf("Preferred vector width in ints: %u\n", vector_width);
   clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, 
         sizeof(vector_width), &vector_width, NULL);      
   printf("Preferred vector width in longs: %u\n", vector_width);
   clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, 
         sizeof(vector_width), &vector_width, NULL);      
   printf("Preferred vector width in floats: %u\n", vector_width);
   clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, 
         sizeof(vector_width), &vector_width, NULL);      
   printf("Preferred vector width in doubles: %u\n", vector_width);
      
#ifndef MAC      
   clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, 
         sizeof(vector_width), &vector_width, NULL);      
   printf("Preferred vector width in halfs: %u\n", vector_width);
#endif
   
   return 0;
}
