package oclia_aparapi;

import com.amd.aparapi.Kernel;

public class AparapiItems {

   public static void main(String[] args) {   
      final int numItems = 8;
      final int numGroups = 4;
      final float[] itemInfo = new float[numItems];

      // Set the kernel code
      Kernel kernel = new Kernel() {
         public void setSizes(int gSize, int lSize) {
            super.setSizes(numItems, numGroups);
         }

         public void run() {
            itemInfo[getGlobalId()] = getGlobalId() * 10.0f + 
                  getGlobalSize() + getLocalId() * 0.1f + getLocalSize() * 0.01f;
         }
      };
      
      // Execute the kernel
      kernel.execute(numItems);
      
      // Display results
      for(int i=0; i<numItems; i++)
    	  System.out.println(itemInfo[i]);
   }
}
