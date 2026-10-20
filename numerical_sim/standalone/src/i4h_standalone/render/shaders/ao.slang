// Verbatim copy of omnisurg/rendering/slang_shaders/omnisurg_ao.slang (152 lines).
struct V2F
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD;
};

Texture2D<float> depth_tex;
Texture2D<float4> source_ao_tex;
SamplerState post_sampler;

uniform float2 output_size;
uniform float2 source_size;
uniform float camera_inv_tan_half_fovy;
uniform float camera_aspect;
uniform float camera_near;
uniform float camera_far;
uniform uint frame_index;
uniform int ao_enabled;
uniform float ao_intensity;
uniform float ao_radius;
uniform float ao_bias;
uniform float ao_power;

[shader("vertex")]
V2F vertex_main(uint vertex_id : SV_VertexID)
{
    V2F output;
    output.uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    output.position = float4(output.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return output;
}

float linear_depth(float depth)
{
    return camera_near * camera_far / max(camera_far - depth * (camera_far - camera_near), 0.0001);
}

float3 reconstruct_view_position(float2 uv, float depth)
{
    float z = linear_depth(depth);
    float2 ndc = uv * 2.0 - 1.0;
    float x = ndc.x * camera_aspect * z / max(camera_inv_tan_half_fovy, 0.0001);
    float y = -ndc.y * z / max(camera_inv_tan_half_fovy, 0.0001);
    return float3(x, y, z);
}

float3 reconstruct_normal(float2 uv, float center_depth)
{
    float2 texel = 1.0 / max(source_size, float2(1.0));
    float3 center = reconstruct_view_position(uv, center_depth);
    float3 right = reconstruct_view_position(uv + float2(texel.x, 0.0), depth_tex.Sample(post_sampler, uv + float2(texel.x, 0.0)));
    float3 up = reconstruct_view_position(uv + float2(0.0, texel.y), depth_tex.Sample(post_sampler, uv + float2(0.0, texel.y)));
    float3 normal = normalize(cross(up - center, right - center));
    return normal.z > 0.0 ? -normal : normal;
}

float hash12(float2 value)
{
    float3 p3 = frac(float3(value.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
}

float2 rotate_dir(float2 dir, float angle)
{
    float s = sin(angle);
    float c = cos(angle);
    return float2(dir.x * c - dir.y * s, dir.x * s + dir.y * c);
}

float occlusion_sample(float2 uv, float2 dir, float center_z, float3 center_pos, float3 normal)
{
    float projected_radius = 0.5 * ao_radius * camera_inv_tan_half_fovy / max(center_z, camera_near);
    projected_radius = min(projected_radius, 0.06);
    float2 sample_uv = uv + dir * projected_radius;
    float sample_depth = depth_tex.Sample(post_sampler, sample_uv);
    if (sample_depth >= 0.9999)
        return 0.0;
    float3 sample_pos = reconstruct_view_position(sample_uv, sample_depth);
    float3 delta = sample_pos - center_pos;
    float distance = length(delta);
    if (distance <= 0.0001)
        return 0.0;
    float hemisphere = saturate((dot(delta, normal) - ao_bias) / max(ao_radius * 0.25, 0.0001));
    float front_depth = saturate((center_z - sample_pos.z - ao_bias) / max(ao_radius * 0.25, 0.0001));
    float recess_depth = saturate((sample_pos.z - center_z - ao_bias) / max(ao_radius * 0.75, 0.0001)) * 0.45;
    float range_weight = 1.0 / (1.0 + distance / max(ao_radius * 0.75, 0.0001));
    return max(max(hemisphere, front_depth), recess_depth) * range_weight;
}

[shader("fragment")]
float4 ao_fragment(V2F input) : SV_Target
{
    if (ao_enabled == 0 || ao_intensity <= 0.0 || ao_radius <= 0.0)
        return float4(1.0);

    float depth = depth_tex.Sample(post_sampler, input.uv);
    if (depth >= 0.9999)
        return float4(1.0);

    float center_z = linear_depth(depth);
    float3 center_pos = reconstruct_view_position(input.uv, depth);
    float3 normal = reconstruct_normal(input.uv, depth);
    float angle = hash12(input.uv * output_size) * 6.2831853;

    float occlusion = 0.0;
    occlusion += occlusion_sample(input.uv, rotate_dir(float2( 1.0,  0.0), angle) * 0.55, center_z, center_pos, normal);
    occlusion += occlusion_sample(input.uv, rotate_dir(float2(-1.0,  0.0), angle) * 0.55, center_z, center_pos, normal);
    occlusion += occlusion_sample(input.uv, rotate_dir(float2( 0.0,  1.0), angle) * 0.55, center_z, center_pos, normal);
    occlusion += occlusion_sample(input.uv, rotate_dir(float2( 0.0, -1.0), angle) * 0.55, center_z, center_pos, normal);
    occlusion += occlusion_sample(input.uv, rotate_dir(normalize(float2( 1.0,  1.0)), angle), center_z, center_pos, normal);
    occlusion += occlusion_sample(input.uv, rotate_dir(normalize(float2(-1.0,  1.0)), angle), center_z, center_pos, normal);
    occlusion += occlusion_sample(input.uv, rotate_dir(normalize(float2( 1.0, -1.0)), angle), center_z, center_pos, normal);
    occlusion += occlusion_sample(input.uv, rotate_dir(normalize(float2(-1.0, -1.0)), angle), center_z, center_pos, normal);
    occlusion += occlusion_sample(input.uv, rotate_dir(float2( 1.0,  0.0), angle) * 1.35, center_z, center_pos, normal);
    occlusion += occlusion_sample(input.uv, rotate_dir(float2(-1.0,  0.0), angle) * 1.35, center_z, center_pos, normal);
    occlusion += occlusion_sample(input.uv, rotate_dir(float2( 0.0,  1.0), angle) * 1.35, center_z, center_pos, normal);
    occlusion += occlusion_sample(input.uv, rotate_dir(float2( 0.0, -1.0), angle) * 1.35, center_z, center_pos, normal);
    occlusion = saturate(occlusion / 6.0);

    float ao = pow(saturate(1.0 - occlusion * ao_intensity), ao_power);
    return float4(ao, ao, ao, 1.0);
}

[shader("fragment")]
float4 blur_fragment(V2F input) : SV_Target
{
    if (ao_enabled == 0 || ao_intensity <= 0.0 || ao_radius <= 0.0)
        return float4(1.0);
    float center_depth = depth_tex.Sample(post_sampler, input.uv);
    float center_z = linear_depth(center_depth);
    float2 texel = 1.0 / max(output_size, float2(1.0));
    float total = 0.0;
    float weight_total = 0.0;

    for (int y = -1; y <= 1; ++y)
    {
        for (int x = -1; x <= 1; ++x)
        {
            float2 offset = float2(float(x), float(y)) * texel;
            float2 sample_uv = input.uv + offset;
            float sample_depth = depth_tex.Sample(post_sampler, sample_uv);
            float sample_z = linear_depth(sample_depth);
            float depth_weight = exp(-abs(sample_z - center_z) * 48.0);
            float kernel_weight = (x == 0 && y == 0) ? 4.0 : ((x == 0 || y == 0) ? 2.0 : 1.0);
            float weight = kernel_weight * depth_weight;
            total += source_ao_tex.Sample(post_sampler, sample_uv).r * weight;
            weight_total += weight;
        }
    }

    float ao = total / max(weight_total, 0.0001);
    return float4(ao, ao, ao, 1.0);
}
