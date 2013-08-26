package oclia_aparapi;

import com.amd.aparapi.Kernel;

public class AparapiRound {

   public static void main(String[] args) {

      final float[] input = new float[]{-6.5f, -3.5f, 3.5f, 6.5f};
      final float[] rintOutput = new float[input.length];			
      final float[] roundOutput = new float[input.length];		
      final float[] ceilOutput = new float[input.length];			
      final float[] floorOutput = new float[input.length];		

      Kernel kernel = new Kernel(){
         public void run() {
            for(int i=0; i<4; i++) {					
               rintOutput[i]  = rint(input[i]);				
               roundOutput[i] = round(input[i]);				
               ceilOutput[i]  = ceil(input[i]);				
               floorOutput[i] = floor(input[i]);				
            }
         }
      };

      kernel.execute(1);							

      System.out.println("rint:  " + rintOutput[0] + ", " + 		
            rintOutput[1] + ", " + rintOutput[2] + ", " + 
            rintOutput[3]);
      System.out.println("round: " + roundOutput[0] + ", " + 
            roundOutput[1] + ", " + roundOutput[2] + ", " + 
            roundOutput[3]);
      System.out.println("ceil:  " + ceilOutput[0] + ", " + 
            ceilOutput[1] + ", " + ceilOutput[2] + ", " + 
            ceilOutput[3]);
      System.out.println("floor: " + floorOutput[0] + ", " + 
            floorOutput[1] + ", " + floorOutput[2] + ", " + 
            floorOutput[3]);
   }
}

