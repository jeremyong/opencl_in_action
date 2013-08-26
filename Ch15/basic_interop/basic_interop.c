#define PROGRAM_FILE "basic_interop.cl"
#define KERNEL_FUNC "basic_interop"
#define VERTEX_SHADER "basic_interop.vert"
#define FRAGMENT_SHADER "basic_interop.frag"

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
GLuint vao[3], vbo[6];
cl_mem mem_objects[6];

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

/* Initialize OpenCl processing */
void init_cl() {

   char *program_buffer, *program_log;
   size_t program_size, log_size;
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

/* Create, compile, and deploy shaders */
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
   glutInitWindowSize(300, 300);
   glutCreateWindow("Basic Interoperability");
   glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

   /* Launch GLEW processing */
   GLenum err = glewInit();
   if (err != GLEW_OK) {
      perror("Couldn't initialize GLEW");
      exit(1);
   }

   /* Create and compile shaders */
   init_shaders();
}

void configure_shared_data() {

   int i, err;

   /* Create 3 vertex array objects - one for each square */
   glGenVertexArrays(3, vao);
   glBindVertexArray(vao[0]);

   /* Create 6 vertex buffer objects (VBOs) - one for each set of coordinates and colors */
   glGenBuffers(6, vbo);

   /* VBO for coordinates of first square */
   glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
   glBufferData(GL_ARRAY_BUFFER, 12*sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); 
   glEnableVertexAttribArray(0);

   /* VBO for colors of first square */
   glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
   glBufferData(GL_ARRAY_BUFFER, 12*sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(1);

   /* VBO for coordinates of second square */
   glBindVertexArray(vao[1]);
   glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
   glBufferData(GL_ARRAY_BUFFER, 12*sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); 
   glEnableVertexAttribArray(0);

   /* VBO for colors of second square */
   glBindBuffer(GL_ARRAY_BUFFER, vbo[3]);
   glBufferData(GL_ARRAY_BUFFER, 12*sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(1);

   /* VBO for coordinates of third square */
   glBindVertexArray(vao[2]);
   glBindBuffer(GL_ARRAY_BUFFER, vbo[4]);
   glBufferData(GL_ARRAY_BUFFER, 12*sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); 
   glEnableVertexAttribArray(0);

   /* VBO for colors of third square */
   glBindBuffer(GL_ARRAY_BUFFER, vbo[5]);
   glBufferData(GL_ARRAY_BUFFER, 12*sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(1);
   glBindBuffer(GL_ARRAY_BUFFER, 0);

   /* Create memory objects from the VBOs */
   for(i=0; i<6; i++) {
      mem_objects[i] = clCreateFromGLBuffer(context, CL_MEM_WRITE_ONLY, 
            vbo[i], &err);
      if(err < 0) {
         perror("Couldn't create a buffer object from the VBO");
         exit(1);
      }
      err = clSetKernelArg(kernel, i, sizeof(cl_mem), &mem_objects[i]);
      if(err < 0) {
         printf("Couldn't set a kernel argument");
         exit(1);   
      }; 
   }
}

void execute_kernel() {

   int err;
   cl_event kernel_event;

   /* Complete OpenGL processing */
   glFinish();

   /* Execute the kernel */
   err = clEnqueueAcquireGLObjects(queue, 6, mem_objects, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't acquire the GL objects");
      exit(1);   
   }

   err = clEnqueueTask(queue, kernel, 0, NULL, &kernel_event);
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      exit(1);   
   }

   err = clWaitForEvents(1, &kernel_event);
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      exit(1);   
   }

   clEnqueueReleaseGLObjects(queue, 6, mem_objects, 0, NULL, NULL);
   clFinish(queue);
   clReleaseEvent(kernel_event);
}

void display(void) {
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   glBindVertexArray(vao[2]);
   glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

   glBindVertexArray(vao[1]);
   glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

   glBindVertexArray(vao[0]);
   glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

   glBindVertexArray(0);
   glutSwapBuffers();
}

void reshape(int w, int h) {
   glViewport(0, 0, (GLsizei)w, (GLsizei)h);
}

int main (int argc, char* argv[]) {
   
   int i;

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
   for(i=0; i<6; i++) {
      clReleaseMemObject(mem_objects[i]);
   }
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);

   /* Deallocate OpenGL resources */
   glDeleteBuffers(6, vbo);

   return 0;
}
