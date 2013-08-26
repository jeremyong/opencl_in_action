__kernel void radix_sort8(__global ushort8 *global_data) {

   typedef union {
      ushort8 vec;
      ushort array[8];
   } vec_array;

   uint one_count, zero_count;
   uint cmp_value = 1;
   vec_array mask, ones, data;

   data.vec = global_data[0];

   /* Rearrange elements according to bits */
   for(int i=0; i<3; i++) {
      zero_count = 0;
      one_count = 0;

      /* Iterate through each element in the input vector */
      for(int j = 0; j < 8; j++) {
         if(data.array[j] & cmp_value)

            /* Place element in ones vector */
            ones.array[one_count++] = data.array[j];
         else {

            /* Increment number of elements with zero */
            mask.array[zero_count++] = j;
         }
      }

      /* Create sorted vector */
      for(int j = zero_count; j < 8; j++)
         mask.array[j] = 8 - zero_count + j;
      data.vec = shuffle2(data.vec, ones.vec, mask.vec);
      cmp_value <<= 1;
   }
   global_data[0] = data.vec;
}
