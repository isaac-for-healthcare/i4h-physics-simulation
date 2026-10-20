// Camera-facing smoke billboards from the smoke system's two vec4 buffers (omnisurg_smoke.slang without
// mesh_vertex_main). Six vertices per particle, no vertex buffer; a dead or expired particle collapses to one
// clip point so the draw always covers the whole pool. Drawn last, over the fluid composite: the hardware depth
// test hides smoke behind opaque geometry and the fluid's smoothed eye depth hides smoke behind the liquid
// (omnisurg drew smoke before the fluid, which then refracted it).
struct SmokeV2F
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float alpha : TEXCOORD1;
    float4 color : TEXCOORD2;
    nointerpolation float eye_depth : TEXCOORD3;
};

uniform float3 camera_pos;
uniform float3 camera_right;
uniform float3 camera_up;
uniform float3 camera_forward;
uniform float camera_inv_tan_half_fovy;
uniform float camera_aspect;
uniform float camera_near;
uniform float camera_far;
uniform float4 mesh_color;
uniform float4 end_color;
Texture2D<float4> diffuse_tex;
SamplerState material_sampler;
StructuredBuffer<float4> smoke_position_age;
StructuredBuffer<float4> smoke_velocity_lifetime;
uniform uint smoke_particle_count;
uniform float start_size;
uniform float end_size;
uniform int has_fluid;
uniform float2 output_size;
Texture2D<float> fluid_depth_tex;

float4 project_position(float3 position)
{
    float3 rel = position - camera_pos;
    float camera_x = dot(rel, camera_right);
    float camera_y = dot(rel, camera_up);
    float camera_z = max(dot(rel, camera_forward), camera_near);

    float clip_x = camera_x * camera_inv_tan_half_fovy / max(camera_aspect, 0.0001);
    float clip_y = camera_y * camera_inv_tan_half_fovy;
    float clip_z = camera_z * camera_far / (camera_far - camera_near)
        - camera_near * camera_far / (camera_far - camera_near);
    return float4(clip_x, clip_y, clip_z, camera_z);
}

float2 smoke_corner_uv(uint corner)
{
    if (corner == 0)
        return float2(0.0, 0.0);
    if (corner == 1)
        return float2(1.0, 0.0);
    if (corner == 2)
        return float2(1.0, 1.0);
    if (corner == 3)
        return float2(0.0, 0.0);
    if (corner == 4)
        return float2(1.0, 1.0);
    return float2(0.0, 1.0);
}

[shader("vertex")]
SmokeV2F vertex_main(uint vertex_id : SV_VertexID)
{
    SmokeV2F output;
    uint particle_id = vertex_id / 6;
    uint corner = vertex_id % 6;
    output.uv = smoke_corner_uv(corner);
    output.alpha = 0.0;
    output.color = mesh_color;
    output.position = float4(0.0, 0.0, 0.0, 1.0);
    output.eye_depth = 0.0;
    if (particle_id >= smoke_particle_count)
        return output;

    float4 position_age = smoke_position_age[particle_id];
    float4 velocity_lifetime = smoke_velocity_lifetime[particle_id];
    float lifetime = velocity_lifetime.w;
    if (lifetime <= 0.0 || position_age.w >= lifetime)
        return output;

    float age_t = saturate(position_age.w / lifetime);
    float size = lerp(start_size, end_size, age_t);
    float2 local = output.uv * 2.0 - 1.0;
    float3 world_position = position_age.xyz
        + camera_right * (local.x * size * 0.5)
        + camera_up * (local.y * size * 0.5);
    output.position = project_position(world_position);
    output.alpha = 1.0;
    output.color = lerp(mesh_color, end_color, age_t);
    output.eye_depth = dot(position_age.xyz - camera_pos, camera_forward);
    return output;
}

[shader("fragment")]
float4 fragment_main(SmokeV2F input) : SV_Target
{
    if (has_fluid != 0)
    {
        int2 max_pixel = int2(max(int(output_size.x), 1), max(int(output_size.y), 1)) - int2(1, 1);
        float fluid_depth = fluid_depth_tex.Load(int3(clamp(int2(input.position.xy), int2(0, 0), max_pixel), 0));
        if (fluid_depth > 0.0 && input.eye_depth > fluid_depth)
            discard;
    }
    float4 sample = diffuse_tex.Sample(material_sampler, input.uv);
    float alpha = sample.a * input.color.a;
    alpha *= input.alpha;
    return float4(saturate(input.color.rgb), alpha);
}
