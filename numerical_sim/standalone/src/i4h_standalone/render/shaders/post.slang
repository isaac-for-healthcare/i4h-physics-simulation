// Post composite. Port of omnisurg/rendering/slang_shaders/omnisurg_post.slang (256 lines). Edits: the pass always
// runs, so postprocess_enabled and its else-branch are gone; the unused depth_tex is dropped; tonemap_operator is
// 0 none (saturate), 1 ACES, 2 Reinhard (omnisurg: 0 ACES, 1 Reinhard); scope_enabled gates the scope mask (the
// canonical omnisurg PostProcessParams has it; with scope_radius 1.5 and the aspect-scaled radius the corners of a
// 16:9 frame would otherwise be black).
struct V2F
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD;
};

Texture2D<float4> scene_color_tex;
Texture2D<float4> bloom_tex;
Texture2D<float4> bloom_global_tex;
Texture2D<float4> ao_tex;
Texture2D<float4> lens_dirt_tex;
Texture2D<float4> auto_exposure_tex;
SamplerState post_sampler;

uniform float2 output_size;
uniform int tonemap_operator;
uniform float exposure;
uniform float3 white_balance;
uniform int auto_exposure_enabled;
uniform int ao_enabled;
uniform int bloom_enabled;
uniform float bloom_intensity;
uniform float bloom_radius;
uniform int lens_dirt_enabled;
uniform float lens_dirt_intensity;
uniform float lens_dirt_threshold;
uniform float lens_dirt_base_opacity;
uniform float lens_dirt_global_drive;
uniform float lens_dirt_mask_gamma;
uniform int lens_distortion_enabled;
uniform float lens_distortion_strength;
uniform float lens_distortion_zoom;
uniform int chromatic_aberration_enabled;
uniform float chromatic_aberration_strength;
uniform int color_grade_enabled;
uniform float color_saturation;
uniform float color_contrast;
uniform float color_gamma;
uniform float color_warmth;
uniform float vignette_strength;
uniform float vignette_radius;
uniform int scope_enabled;
uniform float scope_radius;
uniform float scope_softness;

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

float3 reinhard_tonemap(float3 color)
{
    return color / (1.0 + color);
}

float3 apply_tonemap(float3 color)
{
    if (tonemap_operator == 0)
        return saturate(color);
    if (tonemap_operator == 2)
        return reinhard_tonemap(color);
    return aces_tonemap(color);
}

float luminance(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

float2 centered_lens_uv(float2 uv)
{
    float aspect = output_size.x / max(output_size.y, 1.0);
    float2 centered = uv * 2.0 - 1.0;
    centered.x *= aspect;
    return centered;
}

float radial_distance(float2 uv)
{
    return length(centered_lens_uv(uv));
}

float2 distort_uv(float2 uv)
{
    float aspect = output_size.x / max(output_size.y, 1.0);
    float2 centered = uv * 2.0 - 1.0;
    centered.x *= aspect;
    float r2 = dot(centered, centered);
    float scale = lens_distortion_zoom * (1.0 + lens_distortion_strength * r2);
    centered *= scale;
    centered.x /= aspect;
    return centered * 0.5 + 0.5;
}

float2 radial_unit(float2 uv)
{
    float2 centered = centered_lens_uv(uv);
    float length_centered = length(centered);
    if (length_centered <= 0.0001)
        return float2(0.0);
    float aspect = output_size.x / max(output_size.y, 1.0);
    float2 dir = centered / length_centered;
    dir.x /= aspect;
    return normalize(dir);
}

float endoscope_vignette(float radius)
{
    float start = max(vignette_radius, 0.0001);
    float falloff = saturate((radius - start) / max(1.0 - start, 0.0001));
    return lerp(1.0, 1.0 - vignette_strength, falloff * falloff);
}

float scope_mask(float radius)
{
    if (scope_enabled == 0)
        return 1.0;
    float softness = max(scope_softness, 0.0001);
    return 1.0 - smoothstep(scope_radius - softness, scope_radius, radius);
}

float3 sample_scene_hdr(float2 uv, float effective_exposure)
{
    float ao = ao_enabled != 0 ? saturate(ao_tex.Sample(post_sampler, uv).r) : 1.0;
    return max(scene_color_tex.Sample(post_sampler, uv).rgb, float3(0.0))
        * white_balance
        * effective_exposure
        * ao;
}

float3 sample_bloom_hdr(float2 uv, bool bloom_active, float auto_exposure_multiplier)
{
    if (!bloom_active)
        return float3(0.0);
    return max(bloom_tex.Sample(post_sampler, uv).rgb, float3(0.0)) * bloom_intensity * auto_exposure_multiplier;
}

float3 sample_hdr_with_chromatic_aberration(float2 uv, bool bloom_active, float effective_exposure, float auto_exposure_multiplier)
{
    if (chromatic_aberration_enabled == 0 || chromatic_aberration_strength <= 0.0)
        return sample_scene_hdr(uv, effective_exposure) + sample_bloom_hdr(uv, bloom_active, auto_exposure_multiplier);

    float radius = saturate(radial_distance(uv));
    float2 inv_size = float2(1.0 / max(output_size.x, 1.0), 1.0 / max(output_size.y, 1.0));
    float2 offset = radial_unit(uv) * chromatic_aberration_strength * inv_size * radius * radius;
    float2 red_uv = uv + offset;
    float2 blue_uv = uv - offset;

    float3 red = sample_scene_hdr(red_uv, effective_exposure) + sample_bloom_hdr(red_uv, bloom_active, auto_exposure_multiplier);
    float3 green = sample_scene_hdr(uv, effective_exposure) + sample_bloom_hdr(uv, bloom_active, auto_exposure_multiplier);
    float3 blue = sample_scene_hdr(blue_uv, effective_exposure) + sample_bloom_hdr(blue_uv, bloom_active, auto_exposure_multiplier);
    return float3(red.r, green.g, blue.b);
}

float global_bloom_luminance()
{
    float result = 0.0;
    result = max(result, luminance(max(bloom_global_tex.Sample(post_sampler, float2(0.17, 0.17)).rgb, float3(0.0))));
    result = max(result, luminance(max(bloom_global_tex.Sample(post_sampler, float2(0.50, 0.17)).rgb, float3(0.0))));
    result = max(result, luminance(max(bloom_global_tex.Sample(post_sampler, float2(0.83, 0.17)).rgb, float3(0.0))));
    result = max(result, luminance(max(bloom_global_tex.Sample(post_sampler, float2(0.17, 0.50)).rgb, float3(0.0))));
    result = max(result, luminance(max(bloom_global_tex.Sample(post_sampler, float2(0.50, 0.50)).rgb, float3(0.0))));
    result = max(result, luminance(max(bloom_global_tex.Sample(post_sampler, float2(0.83, 0.50)).rgb, float3(0.0))));
    result = max(result, luminance(max(bloom_global_tex.Sample(post_sampler, float2(0.17, 0.83)).rgb, float3(0.0))));
    result = max(result, luminance(max(bloom_global_tex.Sample(post_sampler, float2(0.50, 0.83)).rgb, float3(0.0))));
    result = max(result, luminance(max(bloom_global_tex.Sample(post_sampler, float2(0.83, 0.83)).rgb, float3(0.0))));
    return result;
}

float3 apply_lens_dirt(float3 color, float2 uv, float3 bloom_drive, float global_drive)
{
    float gamma = max(lens_dirt_mask_gamma, 0.001);
    float3 dirt = pow(saturate(lens_dirt_tex.Sample(post_sampler, uv).rgb), float3(gamma));
    float local_luminance = luminance(bloom_drive);
    float highlight_luminance = max(local_luminance, global_drive * lens_dirt_global_drive);
    float driven_opacity = saturate((highlight_luminance - lens_dirt_threshold) * 2.5);
    float opacity = saturate(lens_dirt_base_opacity + driven_opacity);
    float3 warmed_dirt = dirt * float3(1.0, 0.92, 0.82);
    return color + warmed_dirt * opacity * lens_dirt_intensity;
}

float3 apply_color_grade(float3 color)
{
    float luma = luminance(color);
    color = lerp(float3(luma), color, color_saturation);
    color = (color - 0.5) * color_contrast + 0.5;
    color *= float3(1.0 + color_warmth * 0.08, 1.0, 1.0 - color_warmth * 0.08);
    color = pow(saturate(color), float3(1.0 / max(color_gamma, 0.001)));
    return saturate(color);
}

[shader("fragment")]
float4 fragment_main(V2F input) : SV_Target
{
    float auto_exposure_multiplier = auto_exposure_enabled != 0
        ? max(auto_exposure_tex.Sample(post_sampler, float2(0.5, 0.5)).r, 0.001)
        : 1.0;
    float effective_exposure = exposure * auto_exposure_multiplier;
    bool bloom_active = bloom_enabled != 0 && bloom_intensity > 0.0 && bloom_radius > 0.0;
    bool dirt_active = lens_dirt_enabled != 0 && lens_dirt_intensity > 0.0;

    float2 scene_uv = lens_distortion_enabled != 0 ? distort_uv(input.uv) : input.uv;
    float3 color = sample_hdr_with_chromatic_aberration(
        scene_uv,
        bloom_active,
        effective_exposure,
        auto_exposure_multiplier
    );

    float3 bloom_signal = sample_bloom_hdr(scene_uv, bloom_active, auto_exposure_multiplier);
    float3 bloom_drive = bloom_signal;
    float global_drive = bloom_active ? global_bloom_luminance() * bloom_intensity * auto_exposure_multiplier : 0.0;

    color = apply_tonemap(color);
    if (dirt_active)
        color = apply_lens_dirt(color, input.uv, bloom_drive, global_drive);
    if (color_grade_enabled != 0)
        color = apply_color_grade(color);

    float radius = radial_distance(input.uv);
    color *= endoscope_vignette(radius);
    color *= scope_mask(radius);

    return float4(saturate(color), 1.0);
}
