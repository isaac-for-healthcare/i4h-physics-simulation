// Verbatim copy of omnisurg/rendering/slang_shaders/omnisurg_exposure.slang (99 lines).
struct V2F
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD;
};

Texture2D<float4> scene_color_tex;
Texture2D<float4> bloom_global_tex;
Texture2D<float4> previous_exposure_tex;
SamplerState post_sampler;

uniform float exposure;
uniform float3 white_balance;
uniform float bloom_intensity;
uniform float auto_exposure_target_luma;
uniform float auto_exposure_min;
uniform float auto_exposure_max;
uniform float auto_exposure_speed;
uniform float auto_exposure_highlight_weight;
uniform float auto_exposure_dt;
uniform int auto_exposure_initialized;

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

float scene_luminance_at(float2 uv)
{
    float3 color = max(scene_color_tex.Sample(post_sampler, uv).rgb, float3(0.0));
    color *= white_balance * exposure;
    return luminance(color);
}

float sampled_scene_luminance()
{
    float result = 0.0;
    result += scene_luminance_at(float2(0.20, 0.20));
    result += scene_luminance_at(float2(0.50, 0.20));
    result += scene_luminance_at(float2(0.80, 0.20));
    result += scene_luminance_at(float2(0.20, 0.50));
    result += scene_luminance_at(float2(0.50, 0.50));
    result += scene_luminance_at(float2(0.80, 0.50));
    result += scene_luminance_at(float2(0.20, 0.80));
    result += scene_luminance_at(float2(0.50, 0.80));
    result += scene_luminance_at(float2(0.80, 0.80));
    return result / 9.0;
}

float bloom_luminance_at(float2 uv)
{
    return luminance(max(bloom_global_tex.Sample(post_sampler, uv).rgb, float3(0.0))) * bloom_intensity;
}

float sampled_highlight_luminance()
{
    float result = 0.0;
    result = max(result, bloom_luminance_at(float2(0.17, 0.17)));
    result = max(result, bloom_luminance_at(float2(0.50, 0.17)));
    result = max(result, bloom_luminance_at(float2(0.83, 0.17)));
    result = max(result, bloom_luminance_at(float2(0.17, 0.50)));
    result = max(result, bloom_luminance_at(float2(0.50, 0.50)));
    result = max(result, bloom_luminance_at(float2(0.83, 0.50)));
    result = max(result, bloom_luminance_at(float2(0.17, 0.83)));
    result = max(result, bloom_luminance_at(float2(0.50, 0.83)));
    result = max(result, bloom_luminance_at(float2(0.83, 0.83)));
    return result;
}

[shader("fragment")]
float4 adapt_exposure_fragment(V2F input) : SV_Target
{
    float scene_luma = sampled_scene_luminance();
    float highlight_luma = sampled_highlight_luminance();
    float measured_luma = scene_luma + highlight_luma * auto_exposure_highlight_weight;

    float min_exposure = min(auto_exposure_min, auto_exposure_max);
    float max_exposure = max(auto_exposure_min, auto_exposure_max);
    float target = clamp(
        auto_exposure_target_luma / max(measured_luma, 0.0001),
        min_exposure,
        max_exposure
    );
    float previous = auto_exposure_initialized != 0
        ? max(previous_exposure_tex.Sample(post_sampler, float2(0.5, 0.5)).r, 0.001)
        : 1.0;
    float alpha = saturate(1.0 - exp(-auto_exposure_speed * max(auto_exposure_dt, 0.0)));
    float adapted = lerp(previous, target, alpha);
    return float4(adapted, target, measured_luma, 1.0);
}
