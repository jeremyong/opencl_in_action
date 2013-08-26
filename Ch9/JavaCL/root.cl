__kernel void root(__global float4 *a) {
   int i = get_global_id(0);
   a[i] = sqrt(a[i]);
}
