__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
      CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST; 

__kernel void texture_filter(read_only image2d_t src_image,
                             __global uchar* dst_buffer) {

   int k[9] = {-1, -1, -1, -1, 9, -1, -1, -1, -1};

   int x = get_global_id(0);
   int y = get_global_id(1);

   /* Compute two-dimensional dot product */
   int pixel =
      k[0] * read_imageui(src_image, sampler, (int2)(x-1, y-1)).s0 + 
      k[1] * read_imageui(src_image, sampler, (int2)(x,   y-1)).s0 + 
      k[2] * read_imageui(src_image, sampler, (int2)(x+1, y-1)).s0 + 
      k[3] * read_imageui(src_image, sampler, (int2)(x-1, y)).s0 + 
      k[4] * read_imageui(src_image, sampler, (int2)(x,   y)).s0 + 
      k[5] * read_imageui(src_image, sampler, (int2)(x+1, y)).s0 + 
      k[6] * read_imageui(src_image, sampler, (int2)(x-1, y+1)).s0 + 
      k[7] * read_imageui(src_image, sampler, (int2)(x,   y+1)).s0 + 
      k[8] * read_imageui(src_image, sampler, (int2)(x+1, y+1)).s0;

   /* Set output pixel */
   dst_buffer[y*get_global_size(0) + x] = (uchar)clamp(pixel, 0, 255);
}
