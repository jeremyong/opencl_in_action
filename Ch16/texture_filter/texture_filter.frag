uniform sampler2D tex;
out vec4 new_color;

void main() {
   vec3 color = vec3(texture(tex, gl_TexCoord[0].st));
   new_color = vec4(color, 1.0);
}
