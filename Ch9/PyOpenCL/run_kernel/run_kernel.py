import pyopencl as cl
import numpy

# Create context and command queue
platform = cl.get_platforms()[0]
devices = platform.get_devices()
context = cl.Context(devices)
queue = cl.CommandQueue(context)

# Open program file and build
program_file = open('mult.cl', 'r')
program_text = program_file.read()
program = cl.Program(context, program_text)
try:
   program.build()
except:
   print("Build log:")
   print(program.get_build_info(devices[0], 
         cl.program_build_info.LOG))
   raise

# Create arguments for kernel: a scalar, a LocalMemory object, and a buffer
scalar = numpy.float32(5.0)
lm = cl.LocalMemory(100 * 32)
float_data = numpy.linspace(1, 100, 100).astype(numpy.float32)
float_buffer = cl.Buffer(context, 
      cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, 
      hostbuf=float_data)

# Create, configure, and execute kernel (Seems too easy, doesn't it?)
program.mult(queue, (25,), (25,), scalar, float_buffer, lm)

# Read data back from buffer
cl.enqueue_read_buffer(queue, float_buffer, float_data).wait()

print float_data

