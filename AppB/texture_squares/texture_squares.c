#define VERTEX_SHADER "texture_squares.vert"
#define FRAGMENT_SHADER "texture_squares.frag"

#define NUM_SQUARES 3
#define TEXTURE_1 "checker.png"
#define TEXTURE_2 "square.png"
#define TEXTURE_3 "stripe.png"

#define PNG_DEBUG 3
#include <png.h>

#include <GL/glew.h>
#define FREEGLUT_STATIC
#include <GL/freeglut.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

GLfloat first_coords[] = {-0.15f, -0.15f, 1.0f,
                          -0.15f,  0.15f, 1.0f,
                           0.15f,  0.15f, 1.0f,
                           0.15f, -0.15f, 1.0f};

GLfloat second_coords[] = {-0.30f, -0.30f, 0.0f,
                           -0.30f,  0.30f, 0.0f,
                            0.30f,  0.30f, 0.0f,
                            0.30f, -0.30f, 0.0f};

GLfloat third_coords[] = {-0.45f, -0.45f, -1.0f,
                          -0.45f,  0.45f, -1.0f,
                           0.45f,  0.45f, -1.0f,
                           0.45f, -0.45f, -1.0f};

GLfloat tex_coords[] = {0.0f, 0.0f,
                        0.0f, 1.0f,
                        1.0f, 1.0f,
                        1.0f, 0.0f};

GLuint vao[NUM_SQUARES], vbo[NUM_SQUARES*2], textures[NUM_SQUARES];

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

char* read_file(const char* filename, GLint* size) {

   FILE *handle;
   char *buffer;

   /* Read program file and place content into buffer */
   handle = fopen(filename, "r");
   if(handle == NULL) {
      perror("Couldn't find the file");
      exit(1);
   }
   fseek(handle, 0, SEEK_END);
   *size = ftell(handle);
   rewind(handle);
   buffer = (char*)malloc(*size+1);
   buffer[*size] = '\0';
   fread(buffer, sizeof(char), *size, handle);
   fclose(handle);

   return buffer;
}

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

void init_buffers(void) {

   /* Create a vertex array for each square */
   glGenVertexArrays(NUM_SQUARES, vao);

   /* Create 6 vertex buffer objects (VBOs) - one for each set of coordinates and colors */
   glGenBuffers(NUM_SQUARES * 2, vbo);

   /* VBO for coordinates of first square */
   glBindVertexArray(vao[0]);
   glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
   glBufferData(GL_ARRAY_BUFFER, 12*sizeof(GLfloat), first_coords, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); 
   glEnableVertexAttribArray(0);

   /* VBO for texture coordinates of first square */
   glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
   glBufferData(GL_ARRAY_BUFFER, 8*sizeof(GLfloat), tex_coords, GL_STATIC_DRAW);
   glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(1);

   /* VBO for coordinates of second square */
   glBindVertexArray(vao[1]);
   glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
   glBufferData(GL_ARRAY_BUFFER, 12*sizeof(GLfloat), second_coords, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); 
   glEnableVertexAttribArray(0);

   /* VBO for texture coordinates of second square */
   glBindBuffer(GL_ARRAY_BUFFER, vbo[3]);
   glBufferData(GL_ARRAY_BUFFER, 8*sizeof(GLfloat), tex_coords, GL_STATIC_DRAW);
   glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(1);

   /* VBO for coordinates of third square */
   glBindVertexArray(vao[2]);
   glBindBuffer(GL_ARRAY_BUFFER, vbo[4]);
   glBufferData(GL_ARRAY_BUFFER, 12*sizeof(GLfloat), third_coords, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); 
   glEnableVertexAttribArray(0);

   /* VBO for texture coordinates of third square */
   glBindBuffer(GL_ARRAY_BUFFER, vbo[5]);
   glBufferData(GL_ARRAY_BUFFER, 8*sizeof(GLfloat), tex_coords, GL_STATIC_DRAW);
   glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(1);

   glBindVertexArray(0);
}

void init_shaders(void) {

   GLuint vs, fs, prog;
   char *vs_source, *fs_source;
   GLint vs_length, fs_length;

   vs = glCreateShader(GL_VERTEX_SHADER);
   fs = glCreateShader(GL_FRAGMENT_SHADER);   

   vs_source = read_file(VERTEX_SHADER, &vs_length);
   fs_source = read_file(FRAGMENT_SHADER, &fs_length);

   glShaderSource(vs, 1, (const char**)&vs_source, &vs_length);
   glShaderSource(fs, 1, (const char**)&fs_source, &fs_length);

   compile_shader(vs);
   compile_shader(fs);

   prog = glCreateProgram();

   glBindAttribLocation(prog, 0, "in_coords");
   glBindAttribLocation(prog, 1, "in_texcoords");
      
   glAttachShader(prog, vs);
   glAttachShader(prog, fs);
   
   glLinkProgram(prog);
   glUseProgram(prog);
}

void init_textures(void) {

   png_bytep tex_pixels[NUM_SQUARES];
   size_t width[NUM_SQUARES], height[NUM_SQUARES];
   char *tex_names[NUM_SQUARES] = {TEXTURE_1, TEXTURE_2, TEXTURE_3};
   int i;

   glEnable(GL_TEXTURE_2D);
   glGenTextures(NUM_SQUARES, textures);

   for(i=0; i<NUM_SQUARES; i++) {

      /* Make texture active */
      glBindTexture(GL_TEXTURE_2D, textures[i]);

      /* Set texture parameters */
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      //glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);

      /* Read pixel data and associate it with texture */
      read_image_data(tex_names[i], &tex_pixels[i], &width[i], &height[i]);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width[i], height[i],
         0, GL_RGB, GL_UNSIGNED_BYTE, tex_pixels[i]);
   }
}

void display(void) {
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   glActiveTexture(GL_TEXTURE0);

   glBindVertexArray(vao[2]);
   glBindTexture(GL_TEXTURE_2D, textures[2]);
   glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

   glBindVertexArray(vao[1]);
   glBindTexture(GL_TEXTURE_2D, textures[1]);
   glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

   glBindVertexArray(vao[0]);
   glBindTexture(GL_TEXTURE_2D, textures[0]);
   glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

   glBindVertexArray(0);
   glFinish();
   glutSwapBuffers();
}

void reshape(int w, int h) {
   glViewport(0, 0, (GLsizei)w, (GLsizei)h);
}

int main (int argc, char* argv[]) {
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
   glutInitWindowSize(300, 300);
   glutCreateWindow("Texture Squares");
   glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

   GLenum err = glewInit();
   if (err != GLEW_OK) {
      perror("Couldn't initialize GLEW");
      exit(1);
   }

   init_buffers();
   init_shaders();
   init_textures();

   glutDisplayFunc(display);
   glutReshapeFunc(reshape);   
   glutMainLoop();
   return 0;
}
