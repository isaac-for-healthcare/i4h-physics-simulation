// Verbatim copy of omnisurg/rendering/slang_shaders/omnisurg_bloom.slang (137 lines).
struct V2F
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD;
};

Texture2D<float4> source_tex;
Texture2D<float4> base_tex;
SamplerState post_sampler;

uniform float2 source_size;
uniform float2 output_size;
uniform float exposure;
uniform float3 white_balance;
uniform float bloom_threshold;
uniform float bloom_radius;
uniform int bloom_enabled;  // 0: every level is black (the composite and the exposure ignore them)

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

float karis_weight(float3 color)
{
    return 1.0 / (1.0 + luminance(color));
}

float3 sample_source(float2 uv)
{
    return max(source_tex.Sample(post_sampler, uv).rgb, float3(0.0));
}

float3 sample_hdr_scene(float2 uv)
{
    return sample_source(uv) * white_balance * exposure;
}

float3 soft_threshold(float3 color)
{
    float brightness = max(max(color.r, color.g), color.b);
    float knee = max(bloom_threshold * 0.5, 0.0001);
    float soft = clamp(brightness - bloom_threshold + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 0.0001);
    float contribution = max(brightness - bloom_threshold, soft);
    return color * (contribution / max(brightness, 0.0001));
}

void add_weighted_sample(float3 color, float kernel_weight, inout float3 total, inout float total_weight)
{
    float weight = kernel_weight * karis_weight(color);
    total_weight += weight;
    total += color * weight;
}

float3 downsample_13_tap(float2 uv, bool threshold_first)
{
    float2 texel = 1.0 / max(source_size, float2(1.0));
    float3 total = float3(0.0);
    float total_weight = 0.0;

    float3 a = threshold_first ? soft_threshold(sample_hdr_scene(uv + texel * float2(-2.0, -2.0))) : sample_source(uv + texel * float2(-2.0, -2.0));
    float3 b = threshold_first ? soft_threshold(sample_hdr_scene(uv + texel * float2( 0.0, -2.0))) : sample_source(uv + texel * float2( 0.0, -2.0));
    float3 c = threshold_first ? soft_threshold(sample_hdr_scene(uv + texel * float2( 2.0, -2.0))) : sample_source(uv + texel * float2( 2.0, -2.0));
    float3 d = threshold_first ? soft_threshold(sample_hdr_scene(uv + texel * float2(-2.0,  0.0))) : sample_source(uv + texel * float2(-2.0,  0.0));
    float3 e = threshold_first ? soft_threshold(sample_hdr_scene(uv + texel * float2( 0.0,  0.0))) : sample_source(uv + texel * float2( 0.0,  0.0));
    float3 f = threshold_first ? soft_threshold(sample_hdr_scene(uv + texel * float2( 2.0,  0.0))) : sample_source(uv + texel * float2( 2.0,  0.0));
    float3 g = threshold_first ? soft_threshold(sample_hdr_scene(uv + texel * float2(-2.0,  2.0))) : sample_source(uv + texel * float2(-2.0,  2.0));
    float3 h = threshold_first ? soft_threshold(sample_hdr_scene(uv + texel * float2( 0.0,  2.0))) : sample_source(uv + texel * float2( 0.0,  2.0));
    float3 i = threshold_first ? soft_threshold(sample_hdr_scene(uv + texel * float2( 2.0,  2.0))) : sample_source(uv + texel * float2( 2.0,  2.0));
    float3 j = threshold_first ? soft_threshold(sample_hdr_scene(uv + texel * float2(-1.0, -1.0))) : sample_source(uv + texel * float2(-1.0, -1.0));
    float3 k = threshold_first ? soft_threshold(sample_hdr_scene(uv + texel * float2( 1.0, -1.0))) : sample_source(uv + texel * float2( 1.0, -1.0));
    float3 l = threshold_first ? soft_threshold(sample_hdr_scene(uv + texel * float2(-1.0,  1.0))) : sample_source(uv + texel * float2(-1.0,  1.0));
    float3 m = threshold_first ? soft_threshold(sample_hdr_scene(uv + texel * float2( 1.0,  1.0))) : sample_source(uv + texel * float2( 1.0,  1.0));

    add_weighted_sample(e, 0.125, total, total_weight);

    add_weighted_sample(a, 0.03125, total, total_weight);
    add_weighted_sample(c, 0.03125, total, total_weight);
    add_weighted_sample(g, 0.03125, total, total_weight);
    add_weighted_sample(i, 0.03125, total, total_weight);

    add_weighted_sample(b, 0.0625, total, total_weight);
    add_weighted_sample(d, 0.0625, total, total_weight);
    add_weighted_sample(f, 0.0625, total, total_weight);
    add_weighted_sample(h, 0.0625, total, total_weight);

    add_weighted_sample(j, 0.125, total, total_weight);
    add_weighted_sample(k, 0.125, total, total_weight);
    add_weighted_sample(l, 0.125, total, total_weight);
    add_weighted_sample(m, 0.125, total, total_weight);

    return total / max(total_weight, 0.0001);
}

float3 tent_upsample(float2 uv)
{
    float radius = clamp(bloom_radius / 16.0, 0.5, 4.0);
    float2 texel = radius / max(source_size, float2(1.0));
    float3 total = source_tex.Sample(post_sampler, uv).rgb * 4.0;
    total += source_tex.Sample(post_sampler, uv + texel * float2(-1.0,  0.0)).rgb * 2.0;
    total += source_tex.Sample(post_sampler, uv + texel * float2( 1.0,  0.0)).rgb * 2.0;
    total += source_tex.Sample(post_sampler, uv + texel * float2( 0.0, -1.0)).rgb * 2.0;
    total += source_tex.Sample(post_sampler, uv + texel * float2( 0.0,  1.0)).rgb * 2.0;
    total += source_tex.Sample(post_sampler, uv + texel * float2(-1.0, -1.0)).rgb;
    total += source_tex.Sample(post_sampler, uv + texel * float2( 1.0, -1.0)).rgb;
    total += source_tex.Sample(post_sampler, uv + texel * float2(-1.0,  1.0)).rgb;
    total += source_tex.Sample(post_sampler, uv + texel * float2( 1.0,  1.0)).rgb;
    return max(total / 16.0, float3(0.0));
}

[shader("fragment")]
float4 prefilter_downsample_fragment(V2F input) : SV_Target
{
    if (bloom_enabled == 0)
        return float4(0.0, 0.0, 0.0, 1.0);
    return float4(downsample_13_tap(input.uv, true), 1.0);
}

[shader("fragment")]
float4 downsample_fragment(V2F input) : SV_Target
{
    if (bloom_enabled == 0)
        return float4(0.0, 0.0, 0.0, 1.0);
    return float4(downsample_13_tap(input.uv, false), 1.0);
}

[shader("fragment")]
float4 upsample_fragment(V2F input) : SV_Target
{
    if (bloom_enabled == 0)
        return float4(0.0, 0.0, 0.0, 1.0);
    float3 base = max(base_tex.Sample(post_sampler, input.uv).rgb, float3(0.0));
    return float4(base + tent_upsample(input.uv), 1.0);
}
