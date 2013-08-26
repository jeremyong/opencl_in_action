__kernel void add(__global float *a,
                  __global float *b,
                  __global float *c) {
   
   *c = *a + *b;
}

__kernel void subtract(__global float *a,
                       __global float *b,
                       __global float *c) {
   
   *c = *a - *b;
}

__kernel void multiply(__global float *a,
                       __global float *b,
                       __global float *c) {
   
   *c = *a * *b;
}

__kernel void divide(__global float *a,
                     __global float *b,
                     __global float *c) {
   
   *c = *a / *b;
}
