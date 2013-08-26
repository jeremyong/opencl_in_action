package oclia_javacl;

import java.io.File;
import java.io.IOException;

import com.nativelibs4java.opencl.CLBuildException;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLProgram;
import com.nativelibs4java.opencl.JavaCL;
import com.nativelibs4java.util.IOUtils;

public class JavaCLProgram {

   public static void main(String[] args) {

      CLContext context = JavaCL.createBestContext();

      // Read program file
      String programText = "";
      try {
         programText = IOUtils.readText(new File("root.cl"));
      } catch (IOException e) {
         e.printStackTrace();
      }

      // Create and build program
      CLProgram program = context.createProgram(programText);
      try {
         program.build();
      } catch (CLBuildException e) {
         e.printStackTrace();
      }
   }
}
