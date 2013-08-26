#version 130

in  vec3 in_coords;
in  vec2 in_texcoords;

void main(void) {

   gl_TexCoord[0].st = in_texcoords;
   gl_Position = vec4(in_coords, 1.0);
}
