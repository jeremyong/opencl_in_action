#define _CRT_SECURE_NO_WARNINGS
#define MATRIX_FILE "../bcsstk05.mtx"

#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"

/* Rearrange data to be sorted by row instead of by column */
void sort(int num, int *rows, int *cols, float *values) {

   int i, j, int_swap, index = 0;
   float float_swap;

   for(i=0; i<num; i++) {
      for(j=index; j<num; j++) {
         if(rows[j] == i) {
            if(j == index) {
               index++;
            }
           
            /* Swap row/column/values as necessary */
            else if(j > index) {
               int_swap = rows[index];
               rows[index] = rows[j];
               rows[j] = int_swap;

               int_swap = cols[index];
               cols[index] = cols[j];
               cols[j] = int_swap;

               float_swap = values[index];
               values[index] = values[j];
               values[j] = float_swap;
               index++;
            }
         }
      }
   }
}

int main(int argc, char *argv[]) {

   FILE *mm_handle;
   MM_typecode code;
   int num_rows, num_cols, num_values, i;
   int *rows, *cols;
   float *values;
   double value_double;

   if ((mm_handle = fopen(MATRIX_FILE, "r")) == NULL) {
      perror("Couldn't open the MatrixMarket file");
      exit(1);
   }
   
   /* Print matrix characteristics */
   mm_read_banner(mm_handle, &code);
   if(mm_is_matrix(code))
      printf("This is a matrix.\n");
   else
      printf("This is not a matrix.\n");
   if(mm_is_sparse(code))
      printf("It is sparse, ");
   else
      printf("It is dense, ");
   if(mm_is_complex(code))
      printf("complex-valued, ");
   else
      printf("real-valued, ");
   if(mm_is_symmetric(code))
      printf("and symmetric.\n");
   else
      printf("and not symmetric.\n");

   /* Print matrix dimensions */
   mm_read_mtx_crd_size(mm_handle, &num_rows, &num_cols, &num_values);
   if(mm_is_symmetric(code) || mm_is_skew(code) || mm_is_hermitian(code)) {
      num_values += num_values - num_rows;
   }
   printf("It has %d rows, %d columns, and %d non-zero elements.\n",
         num_rows, num_cols, num_values);

   /* Allocate memory for the values */
   rows = (int*) malloc(num_values * sizeof(int));
   cols = (int*) malloc(num_values * sizeof(int));
   values = (float*) malloc(num_values * sizeof(float));

   /* Read matrix data, sort data, and close file */
   i = 0;
   while(i < num_values) {
      fscanf(mm_handle, "%d %d %lg\n", &rows[i], &cols[i], &value_double);
      values[i] = (float)value_double;
      cols[i]--;
      rows[i]--;
      if((rows[i] != cols[i]) && (mm_is_symmetric(code) || mm_is_skew(code) || mm_is_hermitian(code))) {
         i++;
         rows[i] = cols[i-1];
         cols[i] = rows[i-1];
         values[i] = values[i-1];
      }
      i++;
   }
   sort(num_values, rows, cols, values);
   fclose(mm_handle);

   /* Print all values */
   for(i=0; i<num_values; i++) {
      printf("(%d, %d): %e\n", 
            rows[i], cols[i], values[i]);
   }

   /* Free memory */
   free(rows);
   free(cols);
   free(values);
}
