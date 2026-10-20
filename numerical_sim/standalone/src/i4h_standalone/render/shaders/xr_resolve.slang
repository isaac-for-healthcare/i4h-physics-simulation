// XR eye resolve: the eye's linear HDR scene colour -> a swapchain image (or the mirror window), with the renderer's
// exposure and tonemap (post.slang's operators) and the sRGB encode when the target is not an sRGB format. The XR eyes
// skip the desktop post chain (AO, bloom, auto exposure, lens effects), as omnisurg's XR path does.
struct V2F
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD;
};

Texture2D<float4> source_tex;
SamplerState post_sampler;

uniform float exposure;
uniform int tonemap_operator;
uniform int manual_srgb_encode;

[shader("vertex")]
V2F vertex_main(uint vertex_id : SV_VertexID)
{
    V2F output;
    output.uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    output.position = float4(output.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return output;
}

float3 aces_tonemap(float3 color)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

float3 apply_tonemap(float3 color)
{
    if (tonemap_operator == 0)
        return saturate(color);
    if (tonemap_operator == 2)
        return color / (1.0 + color);
    return aces_tonemap(color);
}

float linear_to_srgb_channel(float value)
{
    value = saturate(value);
    if (value <= 0.0031308)
        return value * 12.92;
    return 1.055 * pow(value, 1.0 / 2.4) - 0.055;
}

[shader("fragment")]
float4 fragment_main(V2F input) : SV_Target
{
    float3 color = apply_tonemap(max(source_tex.Sample(post_sampler, input.uv).rgb, float3(0.0)) * exposure);
    if (manual_srgb_encode != 0)
        color = float3(linear_to_srgb_channel(color.r), linear_to_srgb_channel(color.g), linear_to_srgb_channel(color.b));
    return float4(saturate(color), 1.0);
}
