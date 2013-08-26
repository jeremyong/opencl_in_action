__kernel void string_search(char16 pattern, __global char* text,
     int chars_per_item, __local int* local_result, 
     __global int* global_result) {

   char16 text_vector, check_vector;

   /* initialize local data */
   local_result[0] = 0;
   local_result[1] = 0;
   local_result[2] = 0;
   local_result[3] = 0;

   /* Make sure previous processing has completed */
   barrier(CLK_LOCAL_MEM_FENCE);

   int item_offset = get_global_id(0) * chars_per_item;

   /* Iterate through characters in text */
   for(int i=item_offset; i<item_offset + chars_per_item; i++) {

      /* load global text into private buffer */
      text_vector = vload16(0, text + i);

      /* compare text vector and pattern */
      check_vector = text_vector == pattern;

      /* Check for 'that' */
      if(all(check_vector.s0123))
         atomic_inc(local_result);

      /* Check for 'with' */
      if(all(check_vector.s4567))
         atomic_inc(local_result + 1);

      /* Check for 'have' */
      if(all(check_vector.s89AB))
         atomic_inc(local_result + 2);

      /* Check for 'from' */
      if(all(check_vector.sCDEF))
         atomic_inc(local_result + 3);
   }

   /* Make sure local processing has completed */
   barrier(CLK_GLOBAL_MEM_FENCE);

   /* Perform global reduction */
   if(get_local_id(0) == 0) {
      atomic_add(global_result, local_result[0]);
      atomic_add(global_result + 1, local_result[1]);
      atomic_add(global_result + 2, local_result[2]);
      atomic_add(global_result + 3, local_result[3]);
   }
}
