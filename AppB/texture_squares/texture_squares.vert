#version 130

in  vec3 in_coords;
in  vec2 in_texcoords;

void main(void) {

   gl_TexCoord[0].st = in_texcoords;
   mat3x3 rot_matrix = mat3x3(0.707, 0.641, -0.299,
                             -0.707, 0.641, -0.299,
                             -0.000, 0.423,  0.906);
   vec3 coords = rot_matrix * in_coords;
   gl_Position = vec4(coords, 1.0);
}
