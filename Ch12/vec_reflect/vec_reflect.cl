__kernel void vec_reflect(float4 x_vec, float4 u, 
      __global float4* x_prime) {

   float4 p_mat[4];

   /* Multiply u by sqrt(2)/|u| */
   u *= M_SQRT2_F/length(u); 

   /* Compute Householder matrix */
   p_mat[0] = (float4)(1.0f, 0.0f, 0.0f, 0.0f) - (u * u.x);
   p_mat[1] = (float4)(0.0f, 1.0f, 0.0f, 0.0f) - (u * u.y);
   p_mat[2] = (float4)(0.0f, 0.0f, 1.0f, 0.0f) - (u * u.z); 
   p_mat[3] = (float4)(0.0f, 0.0f, 0.0f, 1.0f) - (u * u.w);

   /* Transform x to obtain x_prime */
   x_prime[0].x = dot(p_mat[0], x_vec);
   x_prime[0].y = dot(p_mat[1], x_vec);
   x_prime[0].z = dot(p_mat[2], x_vec); 
   x_prime[0].w = dot(p_mat[3], x_vec);
}
