import pyopencl as cl

# Create context from devices in first accessible 
platform = cl.get_platforms()[0]
devices = platform.get_devices()
context = cl.Context(devices)

# Create command queue with profiling enabled
queue = cl.CommandQueue(context, devices[0], 
      cl.command_queue_properties.PROFILING_ENABLE)

dev_name = queue.get_info(cl.command_queue_info.DEVICE).\
      get_info(cl.device_info.NAME)
print("Device: %s" % (dev_name))
