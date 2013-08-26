__kernel void basic_interop(__global float4* first_coords, 
      __global float4* first_colors, __global float4* second_coords, 
      __global float4* second_colors, __global float4* third_coords, 
      __global float4* third_colors) {

   first_coords[0] = (float4)(-0.15f, -0.15f,  1.00f, -0.15f);
   first_coords[1] = (float4)( 0.15f,  1.00f,  0.15f,  0.15f);
   first_coords[2] = (float4)( 1.00f,  0.15f, -0.15f,  1.00f);

   first_colors[0] = (float4)(0.00f, 0.00f, 0.00f, 0.25f);
   first_colors[1] = (float4)(0.00f, 0.00f, 0.50f, 0.00f);
   first_colors[2] = (float4)(0.00f, 0.75f, 0.00f, 0.00f);

   second_coords[0] = (float4)(-0.30f, -0.30f,  0.00f, -0.30f);
   second_coords[1] = (float4)( 0.30f,  0.00f,  0.30f,  0.30f);
   second_coords[2] = (float4)( 0.00f,  0.30f, -0.30f,  0.00f);

   second_colors[0] = (float4)(0.00f, 0.00f, 0.00f, 0.00f);
   second_colors[1] = (float4)(0.25f, 0.00f, 0.00f, 0.50f);
   second_colors[2] = (float4)(0.00f, 0.00f, 0.75f, 0.00f);

   third_coords[0] = (float4)(-0.45f, -0.45f, -1.00f, -0.45f);
   third_coords[1] = (float4)( 0.45f, -1.00f,  0.45f,  0.45f);
   third_coords[2] = (float4)(-1.00f,  0.45f, -0.45f, -1.00f);

   third_colors[0] = (float4)(0.00f, 0.00f, 0.00f, 0.00f);
   third_colors[1] = (float4)(0.00f, 0.25f, 0.00f, 0.00f);
   third_colors[2] = (float4)(0.50f, 0.00f, 0.00f, 0.75f);
}
