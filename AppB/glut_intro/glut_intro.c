#define FREEGLUT_STATIC
#include <GL/freeglut.h>
#include <stdio.h>

void reshape(int width, int height) {
   printf("New dimensions: %d %d\n", width, height);
}

void display() {
   glClear(GL_COLOR_BUFFER_BIT);
   printf("Displaying the window\n");
	glutSwapBuffers();
}

int main(int argc, char **argv) {

   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE  );
   glutInitWindowSize(500, 150);
   glutInitWindowPosition(200, 100);
   glutCreateWindow("Introducing GLUT");

   glutDisplayFunc(display);
   glutReshapeFunc(reshape);
   glClearColor(1.0, 1.0, 1.0, 0.0);
	glutMainLoop();
	return 0;
}

