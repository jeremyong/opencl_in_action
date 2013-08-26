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
  cl_device_id *devices;
  cl_uint num_devices, addr_data;
  cl_int i, err;

  /* Extension data */
  char name_data[48], ext_data[4096];

  /* Identify a platform */
  err = clGetPlatformIDs(1, &platform, NULL);
  if(err < 0) {
    perror("Couldn't find any platforms");
    exit(1);
  }

  /* Determine number of connected devices */
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
  if(err < 0) {
    perror("Couldn't find any devices");
    exit(1);
  }

  /* Access connected devices */
  devices = (cl_device_id*)
    malloc(sizeof(cl_device_id) * num_devices);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
                 num_devices, devices, NULL);

  /* Obtain data for each connected device */
  for(i=0; i<num_devices; i++) {
    err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME,
                          sizeof(name_data), name_data, NULL);
    if(err < 0) {
      perror("Couldn't read extension data");
      exit(1);
    }
    clGetDeviceInfo(devices[i], CL_DEVICE_ADDRESS_BITS,
                    sizeof(addr_data), &addr_data, NULL);

    clGetDeviceInfo(devices[i], CL_DEVICE_EXTENSIONS,
                    sizeof(ext_data), ext_data, NULL);

    printf("NAME: %s\nADDRESS_WIDTH: %u\nEXTENSIONS: %s",
           name_data, addr_data, ext_data);
  }

  free(devices);
  return 0;
}
