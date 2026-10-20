// Fullscreen present: the post composite (display-referred, tonemapped) -> a target, with sensor noise and the
// optional manual sRGB encode. Port of omnisurg_present.slang (79 lines); edit: postprocess_enabled dropped (the
// chain always runs). Exposure and the tonemap live in post.slang.
struct V2F
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD;
};

Texture2D<float4> source_tex;
SamplerState post_sampler;

uniform float2 output_size;
uniform uint frame_index;
uniform int sensor_noise_enabled;
uniform float sensor_noise_strength;
uniform float sensor_noise_shadow_boost;
uniform int manual_srgb_encode;

[shader("vertex")]
V2F vertex_main(uint vertex_id : SV_VertexID)
{
    V2F output;
    output.uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    output.position = float4(output.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return output;
}

float luminance(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

float hash12(float2 value)
{
    float3 p3 = frac(float3(value.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
}

float3 apply_sensor_noise(float3 color, float2 uv)
{
    float frame = float(frame_index);
    float2 pixel = uv * output_size;
    float2 seed = pixel + float2(frame * 17.0, frame * 59.0);
    float3 noise = float3(
        hash12(seed),
        hash12(seed + 19.19),
        hash12(seed + 37.73)
    ) - 0.5;
    float shadow_weight = lerp(1.0 + sensor_noise_shadow_boost, 1.0, saturate(luminance(color)));
    return color + noise * sensor_noise_strength * shadow_weight;
}

float linear_to_srgb_channel(float value)
{
    value = saturate(value);
    if (value <= 0.0031308)
        return value * 12.92;
    return 1.055 * pow(value, 1.0 / 2.4) - 0.055;
}

float3 linear_to_srgb(float3 color)
{
    return float3(
        linear_to_srgb_channel(color.r),
        linear_to_srgb_channel(color.g),
        linear_to_srgb_channel(color.b)
    );
}

[shader("fragment")]
float4 fragment_main(V2F input) : SV_Target
{
    float3 color = saturate(source_tex.Sample(post_sampler, input.uv).rgb);
    if (sensor_noise_enabled != 0 && sensor_noise_strength > 0.0)
        color = saturate(apply_sensor_noise(color, input.uv));
    if (manual_srgb_encode != 0)
        color = linear_to_srgb(color);
    return float4(saturate(color), 1.0);
}
