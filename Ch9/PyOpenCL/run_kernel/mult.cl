__kernel void mult(float num, __global float *global_floats,
                   __local float4 *local_floats) {
   
   /* Load vector into local memory */
   int id = get_global_id(0);
   local_floats[id] = vload4(id, global_floats);

   /* Multiply each component by num and store vector to global memory */
   local_floats[id] *= num;
   vstore4(local_floats[id], id, global_floats);
}
