// The screen-space fluid's label and normal into the G-buffer, from the same smoothed depth, occlusion test and
// reconstructed normal the RGB composite uses. Verbatim port of the canonical
// omnisurg/rendering/slang_shaders/omnisurg_recording_fluid.slang (28 lines) over our fluid.slang.
#include "fluid.slang"

uniform uint object_id;
Texture2D<uint> pool_label_tex;  // the nearest pool's label per pixel (fluid_label_fragment); 0 where no pool covers it

struct DatasetFluidOutput
{
    float4 normal_depth : SV_Target0;
    uint label : SV_Target1;
};

struct FluidLabelOut
{
    uint label : SV_Target0;
    float hardware_depth : SV_Depth;
};

// Edit (ours): one sprite pass per pool, depth-tested less_equal against the fluid depth the RGB passes wrote, so every
// pixel keeps the label of the pool whose particle is nearest (omnisurg labelled the whole fluid composite as one).
[shader("fragment")]
FluidLabelOut fluid_label_fragment(FluidDepthV2F input)
{
    FluidLabelOut output;
    float ray_t_near;
    float ray_t_far;
    float3 ray_dir;
    if (!intersect_fluid_sphere(input, ray_t_near, ray_t_far, ray_dir))
        discard;
    float ray_t = max(ray_t_near, 0.0);
    float eye_depth = ray_t * ray_dir.z;
    if (eye_depth <= camera_near)
        discard;
    output.label = object_id;
    output.hardware_depth = hardware_depth_from_eye_depth(eye_depth);
    return output;
}

[shader("fragment")]
DatasetFluidOutput dataset_fluid_fragment(FullscreenV2F input)
{
    int2 pixel = int2(input.uv * output_size);
    float depth = fluid_depth_at_pixel(pixel);
    if (depth <= 0.0)
        discard;
    uint pool = pool_label_tex[pixel];
    if (pool == 0)
        discard;
    float hardware_depth = scene_depth_tex.Sample(post_sampler, input.uv);
    if (hardware_depth < 0.9999 && eye_depth_from_hardware_depth(hardware_depth) <= depth - 0.0005)
        discard;
    float3 normal = reconstruct_fluid_normal(input.uv, pixel, depth);
    DatasetFluidOutput output;
    output.normal_depth = float4(dot(normal, camera_right), dot(normal, camera_up),
        dot(normal, camera_forward), depth);
    output.label = pool;
    return output;
}
