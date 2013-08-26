#define PROGRAM_FILE "texture_filter.cl"
#define KERNEL_FUNC "texture_filter"
#define VERTEX_SHADER "texture_filter.vert"
#define FRAGMENT_SHADER "texture_filter.frag"
#define TEXTURE_FILE "input.png"

#define PNG_DEBUG 3
#include <png.h>

#include <GL/glew.h>
#include <stdio.h>

#ifdef MAC
#include <OpenCL/cl_gl.h>
#include <OpenGL/OpenGL.h>
#include <GLUT/freeglut.h>
#define GL_SHARING_EXTENSION "cl_APPLE_gl_sharing"

#else
#include <CL/cl_gl.h>
#include <GL/freeglut.h>
#include <GL/glx.h>
#define GL_SHARING_EXTENSION "cl_khr_gl_sharing"
#endif

cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_program program;
cl_command_queue queue;
cl_kernel kernel; 
cl_mem in_texture, out_buffer;
GLuint vao, vbo[2], pbo, sampler, texture;
size_t width, height;
png_bytep tex_pixels;

GLfloat vertex_coords[] = {-1.0f, -1.0f, 0.0f,
                           -1.0f,  1.0f, 0.0f,
                            1.0f,  1.0f, 0.0f,
                            1.0f, -1.0f, 0.0f};

GLfloat tex_coords[] = {0.0f, 1.0f,
                        0.0f, 0.0f,
                        1.0f, 0.0f,
                        1.0f, 1.0f};

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

/* Read a character buffer from a file */
char* read_file(const char* filename, size_t* size) {

   FILE *handle;
   char *buffer;

   /* Read program file and place content into buffer */
   handle = fopen(filename, "r");
   if(handle == NULL) {
      perror("Couldn't find the file");
      exit(1);
   }
   fseek(handle, 0, SEEK_END);
   *size = (size_t)ftell(handle);
   rewind(handle);
   buffer = (char*)malloc(*size+1);
   buffer[*size] = '\0';
   fread(buffer, sizeof(char), *size, handle);
   fclose(handle);

   return buffer;
}

/* Initialize OpenCL processing */
void init_cl() {

   char *program_buffer, *program_log;
   size_t program_size, log_size;
   cl_image_format png_format;
   int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't identify a platform");
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

   /* Create OpenCL context properties */
#ifdef MAC
   CGLContextObj mac_context = CGLGetCurrentContext();
   CGLShareGroupObj group = CGLGetShareGroup(mac_context);
   cl_context_properties properties[] = {
      CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, 
      (cl_context_properties)group, 0};
#else 
#ifdef UNIX
   cl_context_properties properties[] = {
      CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(), 
      CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(), 
      CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
#else
   cl_context_properties properties[] = {
      CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(), 
      CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(), 
      CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
#endif
#endif

   /* Create context */
   context = clCreateContext(properties, 1, &device, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);   
   }

   /* Create program from file */
   program_buffer = read_file(PROGRAM_FILE, &program_size);
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
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
            log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   /* Create a command queue */
   queue = clCreateCommandQueue(context, device, 0, &err);
   if(err < 0) {
      perror("Couldn't create a command queue");
      exit(1);   
   };

   /* Create kernel */
   kernel = clCreateKernel(program, KERNEL_FUNC, &err);
   if(err < 0) {
      printf("Couldn't create a kernel: %d", err);
      exit(1);
   };

   /* Read pixel data */
   read_image_data(TEXTURE_FILE, &tex_pixels, &width, &height);

   /* Create the input image object from the PNG data */
   png_format.image_channel_order = CL_R;
   png_format.image_channel_data_type = CL_UNSIGNED_INT8;
   in_texture = clCreateImage2D(context, 
         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
         &png_format, width, height, 0, (void*)tex_pixels, &err);
   if(err < 0) {
      perror("Couldn't create the image object");
      exit(1);
   }; 

   /* Create kernel arguments */
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_texture);
   if(err < 0) {
      printf("Couldn't set a kernel argument");
      exit(1);   
   }; 
}

/* Compile the shader */
void compile_shader(GLint shader) {

   GLint success;
   GLsizei log_size;
   GLchar *log;

   glCompileShader(shader);
   glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
   if (!success) {
      glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_size);
      log = (char*) malloc(log_size+1);
      log[log_size] = '\0';
      glGetShaderInfoLog(shader, log_size+1, NULL, log);
      printf("%s\n", log);
      free(log);
      exit(1);
   }
}

/* Create and configure vertex buffer objects (VBOs) */
void init_buffers(void) {

   /* Create one VAO */
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);

   /* Create one VAO */
   glGenBuffers(2, vbo);

   /* VBO to hold vertex coordinates */
   glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
   glBufferData(GL_ARRAY_BUFFER, 12*sizeof(GLfloat), vertex_coords, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); 
   glEnableVertexAttribArray(0);

   /* VBO to hold texture coordinates */
   glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
   glBufferData(GL_ARRAY_BUFFER, 8*sizeof(GLfloat), tex_coords, GL_STATIC_DRAW);
   glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(1);

   glBindVertexArray(0);
}

void init_textures(void) {

   /* Create texture and make it active*/
   glEnable(GL_TEXTURE_2D);
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);

   /* Set texture parameters */
   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
}

/* Create and compile shaders */
void init_shaders(void) {

   GLuint vs, fs, prog;
   char *vs_source, *fs_source;
   size_t vs_length, fs_length;

   vs = glCreateShader(GL_VERTEX_SHADER);
   fs = glCreateShader(GL_FRAGMENT_SHADER);   

   vs_source = read_file(VERTEX_SHADER, &vs_length);
   fs_source = read_file(FRAGMENT_SHADER, &fs_length);

   glShaderSource(vs, 1, (const char**)&vs_source, (GLint*)&vs_length);
   glShaderSource(fs, 1, (const char**)&fs_source, (GLint*)&fs_length);

   compile_shader(vs);
   compile_shader(fs);

   prog = glCreateProgram();

   glBindAttribLocation(prog, 0, "in_coords");
   glBindAttribLocation(prog, 1, "in_color");

   glAttachShader(prog, vs);
   glAttachShader(prog, fs);

   glLinkProgram(prog);
   glUseProgram(prog);
}


/* Initialize the rendering objects and context properties */
void init_gl(int argc, char* argv[]) {

   /* Initialize the main window */
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
   glutInitWindowSize(233, 90);
   glutCreateWindow("Texture Filter");
   glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

   /* Launch GLEW processing */
   GLenum err = glewInit();
   if (err != GLEW_OK) {
      perror("Couldn't initialize GLEW");
      exit(1);
   }

   /* Create VBOs */
   init_buffers();

   /* Create texture */
   init_textures();

   /* Create and compile shaders */
   init_shaders();
}

void configure_shared_data() {

   int err;

   /* Create and configure pixel buffer */
   glGenBuffers(1, &pbo);
   glBindBuffer(GL_ARRAY_BUFFER, pbo);
   glBufferData(GL_ARRAY_BUFFER, width*height*sizeof(char), 
         NULL, GL_STATIC_DRAW);
   glBindBuffer(GL_ARRAY_BUFFER, 0);

   out_buffer = clCreateFromGLBuffer(context, CL_MEM_WRITE_ONLY, 
         pbo, &err);
   if(err < 0) {
      perror("Couldn't create a buffer object from the PBO");
      exit(1);
   }

   err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_buffer);
   if(err < 0) {
      printf("Couldn't set a kernel argument");
      exit(1);   
   };
}

void execute_kernel() {

   int err;
   cl_event kernel_event;
   size_t global_size[2];

   /* Complete OpenGL processing */
   glFinish();

   /* Acquire exlusive access to OpenGL buffer */
   err = clEnqueueAcquireGLObjects(queue, 1, &out_buffer, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't acquire the GL object");
      exit(1);   
   }

   /* Execute the kernel */
   global_size[0] = width;
   global_size[1] = height;  
   err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, NULL, 
         0, NULL, &kernel_event);
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      exit(1);   
   }

   /* Wait for kernel execution to complete */
   err = clWaitForEvents(1, &kernel_event);
   if(err < 0) {
      perror("Couldn't wait for events");
      exit(1);   
   }

   clEnqueueReleaseGLObjects(queue, 1, &out_buffer, 0, NULL, NULL);
   clFinish(queue);
   clReleaseEvent(kernel_event);

   /* Copy pixels from pbo to texture */
   glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height,
      0, GL_LUMINANCE, GL_UNSIGNED_BYTE, 0);
   glActiveTexture(GL_TEXTURE0);
}

void display(void) {
   glClear(GL_COLOR_BUFFER_BIT);
   glBindVertexArray(vao);
   glBindTexture(GL_TEXTURE_2D, texture);
   glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
   glBindVertexArray(0);
   glutSwapBuffers();
}

void reshape(int w, int h) {
   glViewport(0, 0, (GLsizei)w, (GLsizei)h);
}

int main (int argc, char* argv[]) {

   /* Start GL processing */
   init_gl(argc, argv);

   /* Initialize CL data structures */
   init_cl();

   /* Create CL and GL data objects */
   configure_shared_data();

   /* Execute kernel */
   execute_kernel();

   /* Set callback functions */
   glutDisplayFunc(display);
   glutReshapeFunc(reshape);   

   /* Start processing loop */
   glutMainLoop();

   /* Deallocate OpenCL resources */
   clReleaseMemObject(in_texture);
   clReleaseMemObject(out_buffer);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);

   /* Deallocate OpenGL resources */
   glDeleteBuffers(2, vbo);

   return 0;
}
