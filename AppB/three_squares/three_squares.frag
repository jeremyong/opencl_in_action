in  vec3 new_color;
out vec4 out_color;

void main(void) {
	vec3 tmp_color = new_color + vec3(0.25f, 0.25f, 0.25f);
	out_color = vec4(tmp_color, 1.0);
}
