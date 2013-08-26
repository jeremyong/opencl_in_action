__kernel void sphere(__global float4* vertices, float tick) {

   int longitude = get_global_id(0)/16;
   int latitude = get_global_id(0) % 16;

   float sign = -2.0f * (longitude % 2) + 1.0f;
   float phi = 2.0f * M_PI_F * longitude/16 + tick;
   float theta = M_PI_F * latitude/16;

   vertices[get_global_id(0)].x = RADIUS * sin(theta) * cos(phi);
   vertices[get_global_id(0)].y = RADIUS * sign * cos(theta);
   vertices[get_global_id(0)].z = RADIUS * sin(theta) * sin(phi);
   vertices[get_global_id(0)].w = 1.0f;
}
