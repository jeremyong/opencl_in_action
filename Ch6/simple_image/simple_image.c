#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "simple_image.cl"
#define KERNEL_FUNC "simple_image"

#define PNG_DEBUG 3
#include <png.h>

#define INPUT_FILE "blank.png"
#define OUTPUT_FILE "output.png"

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

void read_image_data(const char* filename, png_bytep* data, size_t* w, size_t* h) {

   int i;

   /* Open input file */
   FILE *png_input;
   if((png_input = fopen(filename, "rb")) == NULL) {
      perror("Can't read input image file");
      exit(1);
   }

   /* Read image data */
   png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
   png_infop info_ptr = png_create_info_struct(png_ptr);
   png_init_io(png_ptr, png_input);
   png_read_info(png_ptr, info_ptr);
   *w = png_get_image_width(png_ptr, info_ptr);
   *h = png_get_image_height(png_ptr, info_ptr);

   /* Allocate memory and read image data */
   *data = malloc(*h * png_get_rowbytes(png_ptr, info_ptr));
   for(i=0; i<*h; i++) {
      png_read_row(png_ptr, *data + i * png_get_rowbytes(png_ptr, info_ptr), NULL);
   }

   /* Close input file */
   png_read_end(png_ptr, info_ptr);
   png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
   fclose(png_input);
}

void write_image_data(const char* filename, png_bytep data, size_t w, size_t h) {

   int i;

   /* Open output file */
   FILE *png_output;
   if((png_output = fopen(filename, "wb")) == NULL) {
      perror("Create output image file");
      exit(1);
   }

   /* Write image data */
   png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
   png_infop info_ptr = png_create_info_struct(png_ptr);
   png_init_io(png_ptr, png_output);
   png_set_IHDR(png_ptr, info_ptr, w, h, 16,
         PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
         PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
   png_write_info(png_ptr, info_ptr);
   for(i=0; i<h; i++) {
      png_write_row(png_ptr, data + i*png_get_rowbytes(png_ptr, info_ptr));
   }

   /* Close file */
   png_write_end(png_ptr, NULL);
   png_destroy_write_struct(&png_ptr, &info_ptr);
   fclose(png_output);
}

int main(int argc, char **argv) {

   /* Host/device data structures */
   cl_device_id device;
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel kernel;
   cl_int err;
   size_t global_size[2];

   /* Image data */
   png_bytep pixels;
   cl_image_format png_format;
   cl_mem input_image, output_image;
   size_t origin[3], region[3];
   size_t width, height;

   /* Open input file and read image data */
   read_image_data(INPUT_FILE, &pixels, &width, &height);

   /* Create a device and context */
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
      printf("Couldn't create a kernel: %d", err);
      exit(1);
   };

   /* Create image object */
   png_format.image_channel_order = CL_LUMINANCE;
   png_format.image_channel_data_type = CL_UNORM_INT16;
   input_image = clCreateImage2D(context, 
         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
         &png_format, width, height, 0, (void*)pixels, &err);
   output_image = clCreateImage2D(context, 
         CL_MEM_WRITE_ONLY, &png_format, width, height, 0, NULL, &err);
   if(err < 0) {
      perror("Couldn't create the image object");
      exit(1);
   }; 

   /* Create kernel arguments */
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);
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
   global_size[0] = height; global_size[1] = width;
   err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, 
         NULL, 0, NULL, NULL);  
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      exit(1);
   }

   /* Read the image object */
   origin[0] = 0; origin[1] = 0; origin[2] = 0;
   region[0] = width; region[1] = height; region[2] = 1;
   err = clEnqueueReadImage(queue, output_image, CL_TRUE, origin, 
         region, 0, 0, (void*)pixels, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't read from the image object");
      exit(1);   
   }

   /* Create output PNG file and write data */
   write_image_data(OUTPUT_FILE, pixels, width, height);

   /* Deallocate resources */
   free(pixels);
   clReleaseMemObject(input_image);
   clReleaseMemObject(output_image);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}
