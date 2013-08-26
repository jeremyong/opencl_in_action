package oclia_javacl;

import java.io.File;
import java.nio.FloatBuffer;

import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLFloatBuffer;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLProgram;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.util.IOUtils;
import com.nativelibs4java.util.NIOUtils;

public class JavaCLRoot {

   public static final int NUM_FLOATS = 64;
   public static final int NUM_ITEMS = NUM_FLOATS/4;
	
   public static void main(String[] args) throws Exception {
	   
      // Create context, queue
      CLContext context = JavaCL.createBestContext();
      CLQueue queue = context.createDefaultQueue();

      // Initialize data buffer
      FloatBuffer dataBuffer = NIOUtils.directFloats(NUM_FLOATS, context.getByteOrder());
      for(int i = 0; i < NUM_FLOATS; i++) {
         dataBuffer.put(i, i * 5.0f);
      }
      CLFloatBuffer buff = context.createFloatBuffer(Usage.InputOutput, dataBuffer, true);

      // Create and enqueue kernel
      String programText = IOUtils.readText(new File("root.cl"));
      CLProgram program = context.createProgram(programText);
      CLKernel kernel = program.createKernel("root", buff);
      CLEvent kernelEvent = kernel.enqueueNDRange(queue, new int[]{NUM_ITEMS}, new int[]{NUM_ITEMS});

      // Read and verify output
      buff.read(queue, dataBuffer, true, kernelEvent);
      for(int i = 0; i < NUM_FLOATS; i++)
         System.out.println(i + ": " + dataBuffer.get(i));
   }
}
