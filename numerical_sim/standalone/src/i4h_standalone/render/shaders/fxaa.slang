// Port of omnisurg/rendering/slang_shaders/omnisurg_fxaa.slang (69 lines). Edit: fxaa_enabled passthrough, so the pass
// runs every frame and the toggle is a uniform.
struct V2F
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD;
};

Texture2D<float4> source_tex;
SamplerState post_sampler;

uniform float2 output_size;
uniform int fxaa_enabled;
uniform float fxaa_subpix;
uniform float fxaa_edge_threshold;
uniform float fxaa_edge_threshold_min;

[shader("vertex")]
V2F vertex_main(uint vertex_id : SV_VertexID)
{
    V2F output;
    output.uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    output.position = float4(output.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return output;
}

float luma(float3 color)
{
    return dot(color, float3(0.299, 0.587, 0.114));
}

[shader("fragment")]
float4 fragment_main(V2F input) : SV_Target
{
    float2 texel = 1.0 / max(output_size, float2(1.0));
    float3 rgb_m = source_tex.Sample(post_sampler, input.uv).rgb;
    if (fxaa_enabled == 0)
        return float4(saturate(rgb_m), 1.0);
    float3 rgb_nw = source_tex.Sample(post_sampler, input.uv + texel * float2(-1.0, -1.0)).rgb;
    float3 rgb_ne = source_tex.Sample(post_sampler, input.uv + texel * float2( 1.0, -1.0)).rgb;
    float3 rgb_sw = source_tex.Sample(post_sampler, input.uv + texel * float2(-1.0,  1.0)).rgb;
    float3 rgb_se = source_tex.Sample(post_sampler, input.uv + texel * float2( 1.0,  1.0)).rgb;

    float luma_m = luma(rgb_m);
    float luma_nw = luma(rgb_nw);
    float luma_ne = luma(rgb_ne);
    float luma_sw = luma(rgb_sw);
    float luma_se = luma(rgb_se);
    float luma_min = min(luma_m, min(min(luma_nw, luma_ne), min(luma_sw, luma_se)));
    float luma_max = max(luma_m, max(max(luma_nw, luma_ne), max(luma_sw, luma_se)));
    float contrast = luma_max - luma_min;
    float threshold = max(fxaa_edge_threshold_min, luma_max * fxaa_edge_threshold);
    if (contrast < threshold)
        return float4(saturate(rgb_m), 1.0);

    float2 dir;
    dir.x = -((luma_nw + luma_ne) - (luma_sw + luma_se));
    dir.y =  ((luma_nw + luma_sw) - (luma_ne + luma_se));
    float dir_reduce = max((luma_nw + luma_ne + luma_sw + luma_se) * (0.25 * 0.5 * fxaa_subpix), 1.0 / 128.0);
    float rcp_dir_min = 1.0 / (min(abs(dir.x), abs(dir.y)) + dir_reduce);
    dir = clamp(dir * rcp_dir_min, float2(-8.0), float2(8.0)) * texel;

    float3 rgb_a = 0.5 * (
        source_tex.Sample(post_sampler, input.uv + dir * (1.0 / 3.0 - 0.5)).rgb +
        source_tex.Sample(post_sampler, input.uv + dir * (2.0 / 3.0 - 0.5)).rgb);
    float3 rgb_b = rgb_a * 0.5 + 0.25 * (
        source_tex.Sample(post_sampler, input.uv + dir * -0.5).rgb +
        source_tex.Sample(post_sampler, input.uv + dir * 0.5).rgb);

    float luma_b = luma(rgb_b);
    float3 resolved = (luma_b < luma_min || luma_b > luma_max) ? rgb_a : rgb_b;
    resolved = lerp(rgb_m, resolved, saturate(fxaa_subpix));
    return float4(saturate(resolved), 1.0);
}
