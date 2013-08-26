import pyopencl as cl

# Create context from devices in first accessible platform
platform = cl.get_platforms()[0]
devices = platform.get_devices()
context = cl.Context(devices)

# Create program from arith.cl file
program_file = open('arith.cl', 'r')
program_text = program_file.read()
program = cl.Program(context, program_text)

# Build program and print log in the event of an error
try:
   program.build()
except:
   print("Build log:")
   print(program.get_build_info(devices[0], 
         cl.program_build_info.LOG))
   raise

# Create kernel from 'add' function
add_kernel = cl.Kernel(program, 'add')

# Create kernel from 'multiply' function
mult_kernel = program.multiply

print("Kernel Name:"),
print(mult_kernel.get_info(cl.kernel_info.FUNCTION_NAME))
