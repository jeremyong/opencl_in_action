#define VERTEX_SHADER "three_squares.vert"
#define FRAGMENT_SHADER "three_squares.frag"

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
GLfloat first_colors[] = {0.0f,  0.0f, 0.0f, 
                          0.25f, 0.0f, 0.0f, 
                          0.50f, 0.0f, 0.0f, 
                          0.75f, 0.0f, 0.0f};

GLfloat second_coords[] = {-0.30f, -0.30f, 0.0f,
                           -0.30f,  0.30f, 0.0f,
                            0.30f,  0.30f, 0.0f,
                            0.30f, -0.30f, 0.0f};
GLfloat second_colors[] = {0.0f, 0.0f,  0.0f, 
                           0.0f, 0.25f, 0.0f, 
                           0.0f, 0.50f,  0.0f, 
                           0.0f, 0.75f, 0.0f};

GLfloat third_coords[] = {-0.45f, -0.45f, -1.0f,
                          -0.45f,  0.45f, -1.0f,
                           0.45f,  0.45f, -1.0f,
                           0.45f, -0.45f, -1.0f};
GLfloat third_colors[] = {0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.25f, 
                          0.0f, 0.0f, 0.50f, 
                          0.0f, 0.0f, 0.75f};

GLuint vao[3], vbo[6];

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

   /* Create 3 vertex array objects - one for each square */
   glGenVertexArrays(3, vao);

   /* Create 6 vertex buffer objects (VBOs) - one for each set of coordinates and colors */
   glGenBuffers(6, vbo);

   /* VBO for coordinates of first square */
   glBindVertexArray(vao[0]);
   glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
   glBufferData(GL_ARRAY_BUFFER, 12*sizeof(GLfloat), first_coords, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); 
   glEnableVertexAttribArray(0);

   /* VBO for colors of first square */
   glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
   glBufferData(GL_ARRAY_BUFFER, 12*sizeof(GLfloat), first_colors, GL_STATIC_DRAW);
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(1);

   /* VBO for coordinates of second square */
   glBindVertexArray(vao[1]);
   glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
   glBufferData(GL_ARRAY_BUFFER, 12*sizeof(GLfloat), second_coords, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); 
   glEnableVertexAttribArray(0);

   /* VBO for colors of second square */
   glBindBuffer(GL_ARRAY_BUFFER, vbo[3]);
   glBufferData(GL_ARRAY_BUFFER, 12*sizeof(GLfloat), second_colors, GL_STATIC_DRAW);
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(1);

   /* VBO for coordinates of third square */
   glBindVertexArray(vao[2]);
   glBindBuffer(GL_ARRAY_BUFFER, vbo[4]);
   glBufferData(GL_ARRAY_BUFFER, 12*sizeof(GLfloat), third_coords, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); 
   glEnableVertexAttribArray(0);

   /* VBO for colors of third square */
   glBindBuffer(GL_ARRAY_BUFFER, vbo[5]);
   glBufferData(GL_ARRAY_BUFFER, 12*sizeof(GLfloat), third_colors, GL_STATIC_DRAW);
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
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
   glBindAttribLocation(prog, 1, "in_color");
      
   glAttachShader(prog, vs);
   glAttachShader(prog, fs);
   
   glLinkProgram(prog);
   glUseProgram(prog);
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
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
   glutInitWindowSize(300, 300);
   glutCreateWindow("Three Squares");
   glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

   GLenum err = glewInit();
   if (err != GLEW_OK) {
      perror("Couldn't initialize GLEW");
      exit(1);
   }

   init_buffers();
   init_shaders();
   glutDisplayFunc(display);
   glutReshapeFunc(reshape);   
   glutMainLoop();
   return 0;
}
