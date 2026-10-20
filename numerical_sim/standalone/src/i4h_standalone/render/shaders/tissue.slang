// Verbatim copy of omnisurg/rendering/slang_shaders/omnisurg_tissue.slang (470 lines).
struct V2F
{
    float4 position : SV_Position;
    float3 world_position : WORLD_POSITION;
    float3 world_normal : NORMAL;
    float4 world_tangent : TANGENT;
    float2 uv : TEXCOORD;
    float4 vertex_color : COLOR;
};

uniform float3 camera_pos;
uniform float3 camera_right;
uniform float3 camera_up;
uniform float3 camera_forward;
uniform float camera_inv_tan_half_fovy;
uniform float camera_aspect;
uniform float camera_near;
uniform float camera_far;
uniform float4x4 camera_view_projection;  // world -> clip (Camera.view_projection; an XR eye passes its off-axis one)
uniform float4 mesh_color;
uniform float3 light_key_dir;
uniform float3 light_fill_dir;
uniform float3 light_back_dir;
uniform float3 light_key_color;
uniform float3 light_fill_color;
uniform float3 light_ambient_color;
uniform float3 light_rim_color;
uniform int endoscopic_enabled;
uniform float headlight_intensity;
uniform float headlight_range;
uniform float headlight_inner_cos;
uniform float headlight_outer_cos;
uniform float directional_fill_scale;
uniform int key_light_enabled;
uniform int fill_light_enabled;
uniform int back_light_enabled;
uniform int endoscope_light_enabled;
uniform float3 endoscope_light_color;
uniform int has_environment_map;
uniform int environment_lighting_enabled;
uniform float environment_intensity;
uniform float environment_rotation;
uniform float environment_pitch;

Texture2D<float4> diffuse_base_tex;
Texture2D<float4> diffuse_damage_tex;
Texture2D<float4> diffuse_coag_tex;
Texture2D<float4> diffuse_blood_tex;
Texture2D<float4> normal_base_tex;
Texture2D<float4> normal_damage_tex;
Texture2D<float4> normal_coag_tex;
Texture2D<float4> normal_blood_tex;
Texture2D<float4> spec_base_tex;
Texture2D<float4> spec_damage_tex;
Texture2D<float4> spec_coag_tex;
Texture2D<float4> spec_blood_tex;
Texture2D<float4> layer_mask_damage_tex;
Texture2D<float4> layer_mask_coag_tex;
Texture2D<float4> layer_mask_blood_tex;
Texture2D<float4> blood_mask_tex;
Texture2D<float4> heat_mask_tex;
Texture2D<float4> environment_map;
SamplerState material_sampler;
SamplerState environment_sampler;

#ifndef OMNISURG_TISSUE_FRAGMENT_ENTRY
#define OMNISURG_TISSUE_FRAGMENT_ENTRY fragment_main
#endif

uniform float normal_strength;
uniform float specular_scale;
uniform float roughness_bias;
uniform float ambient;
uniform float rim_strength;
uniform float coat_weight;
uniform float coat_ior;
uniform float color_gain;
uniform float blend_damage;
uniform float blend_coag;
uniform float blend_blood;
uniform int debug_mode;
uniform float wetness;
uniform float wet_spec_scale;
uniform float wet_roughness;
uniform float3 subsurface_color;
uniform float subsurface_strength;
uniform float blood_wetness;
uniform int specular_aa_enabled;
uniform float specular_aa_strength;
uniform float specular_aa_min_roughness;

static const int DEBUG_FINAL = 0;
static const int DEBUG_VERTEX_BLEND_RGB = 1;
static const int DEBUG_MASKED_LAYER_WEIGHTS = 2;
static const int DEBUG_BLENDED_DIFFUSE = 3;
static const int DEBUG_BLENDED_NORMAL = 4;
static const int DEBUG_SPEC_ROUGHNESS = 5;
static const int DEBUG_HEAT_BLOOD_MASKS = 6;
static const int DEBUG_RESERVED_MATERIAL_PARAMS = 99;
static const float PI = 3.14159265;

[shader("vertex")]
V2F vertex_main(
    float3 position : POSITION,
    float3 normal : NORMAL,
    float4 tangent : TANGENT,
    float2 uv : TEXCOORD,
    float4 color : COLOR)
{
    V2F output;

    output.position = mul(camera_view_projection, float4(position, 1.0));
    output.world_position = position;
    output.world_normal = normal;
    output.world_tangent = tangent;
    output.uv = uv;
    output.vertex_color = color;
    return output;
}

float3 normalize_or(float3 value, float3 fallback)
{
    float length_value = length(value);
    return length_value > 0.000001 ? value / length_value : fallback;
}

struct KeyLightSample
{
    float3 direction;
    float attenuation;
};

float directional_fill_multiplier()
{
    return endoscopic_enabled != 0 ? saturate(directional_fill_scale) : 1.0;
}

KeyLightSample compute_headlight_key_light(float3 world_position)
{
    KeyLightSample sample;
    if (endoscopic_enabled == 0)
    {
        sample.direction = normalize_or(light_key_dir, float3(0.0, 0.0, 1.0));
        sample.attenuation = 1.0;
        return sample;
    }

    float3 to_light = camera_pos - world_position;
    float distance_to_light = length(to_light);
    sample.direction = normalize_or(to_light, -camera_forward);
    float spot_cos = dot(-sample.direction, normalize_or(camera_forward, float3(0.0, 0.0, -1.0)));
    float spot = smoothstep(headlight_outer_cos, max(headlight_inner_cos, headlight_outer_cos + 0.0001), spot_cos);
    float range_ratio = distance_to_light / max(headlight_range, 0.0001);
    float inverse_square = 1.0 / (1.0 + range_ratio * range_ratio);
    float range_window = saturate(1.0 - range_ratio * range_ratio);
    sample.attenuation = max(headlight_intensity, 0.0) * inverse_square * range_window * spot;
    return sample;
}

float3 fallback_tangent(float3 normal)
{
    float3 axis = abs(normal.y) < 0.95 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    return normalize_or(cross(axis, normal), float3(1.0, 0.0, 0.0));
}

float3 blend_rgb_layers(float2 uv, float damage_blend, float coag_blend, float blood_blend)
{
    float3 value = diffuse_base_tex.Sample(material_sampler, uv).rgb;
    value = lerp(value, diffuse_damage_tex.Sample(material_sampler, uv).rgb, damage_blend);
    value = lerp(value, diffuse_coag_tex.Sample(material_sampler, uv).rgb, coag_blend);
    value = lerp(value, diffuse_blood_tex.Sample(material_sampler, uv).rgb, blood_blend);
    return value;
}

float3 blend_normal_layers(float2 uv, float damage_blend, float coag_blend, float blood_blend)
{
    float3 value = normal_base_tex.Sample(material_sampler, uv).xyz * 2.0 - 1.0;
    float3 damage_normal = normal_damage_tex.Sample(material_sampler, uv).xyz * 2.0 - 1.0;
    float3 coag_normal = normal_coag_tex.Sample(material_sampler, uv).xyz * 2.0 - 1.0;
    float3 blood_normal = normal_blood_tex.Sample(material_sampler, uv).xyz * 2.0 - 1.0;
    value = lerp(value, damage_normal, damage_blend);
    value = lerp(value, coag_normal, coag_blend);
    value = lerp(value, blood_normal, blood_blend);
    return value;
}

float blend_spec_layers(float2 uv, float damage_blend, float coag_blend, float blood_blend)
{
    float value = spec_base_tex.Sample(material_sampler, uv).r;
    value = lerp(value, spec_damage_tex.Sample(material_sampler, uv).r, damage_blend);
    value = lerp(value, spec_coag_tex.Sample(material_sampler, uv).r, coag_blend);
    value = lerp(value, spec_blood_tex.Sample(material_sampler, uv).r, blood_blend);
    return value;
}

float apply_specular_aa(float roughness, float3 normal)
{
    if (specular_aa_enabled == 0 || specular_aa_strength <= 0.0)
        return roughness;

    float3 normal_dx = ddx(normal);
    float3 normal_dy = ddy(normal);
    float variance = dot(normal_dx, normal_dx) + dot(normal_dy, normal_dy);
    float boost = saturate(variance * specular_aa_strength);
    return saturate(max(roughness, specular_aa_min_roughness) + boost);
}

float3 fresnel_schlick(float cos_theta, float3 f0)
{
    float factor = pow(1.0 - saturate(cos_theta), 5.0);
    return f0 + (float3(1.0) - f0) * factor;
}

float clearcoat_fresnel_omnisurface(float n_dot, float ior)
{
    float nr_base = (ior - 1.0) / (ior + 1.0);
    float nr = nr_base * nr_base;
    return lerp(nr, 1.0, pow(1.0 - saturate(n_dot), 5.0));
}

float ggx_distribution(float n_dot_h, float roughness)
{
    float alpha = max(roughness * roughness, 0.001);
    float alpha_squared = alpha * alpha;
    float denominator = n_dot_h * n_dot_h * (alpha_squared - 1.0) + 1.0;
    return alpha_squared / max(PI * denominator * denominator, 0.001);
}

float smith_ggx_geometry(float n_dot_l, float n_dot_v, float roughness)
{
    float k = (roughness + 1.0) * (roughness + 1.0) * 0.125;
    float geometry_l = n_dot_l / max(n_dot_l * (1.0 - k) + k, 0.001);
    float geometry_v = n_dot_v / max(n_dot_v * (1.0 - k) + k, 0.001);
    return geometry_l * geometry_v;
}

float2 environment_equirect_uv(float3 direction)
{
    float3 dir = normalize_or(direction, float3(0.0, 1.0, 0.0));
    float pitch_sin = sin(environment_pitch);
    float pitch_cos = cos(environment_pitch);
    dir = normalize_or(
        float3(
            dir.x,
            dir.y * pitch_cos - dir.z * pitch_sin,
            dir.y * pitch_sin + dir.z * pitch_cos
        ),
        dir
    );
    float u = atan2(dir.z, dir.x) * 0.159154943 + environment_rotation;
    float v = acos(clamp(dir.y, -1.0, 1.0)) * 0.318309886;
    return float2(frac(u), v);
}

float3 sample_environment_uv(float2 uv)
{
    if (has_environment_map == 0 || environment_lighting_enabled == 0)
    {
        return float3(0.0);
    }
    float2 wrapped_uv = float2(frac(uv.x), clamp(uv.y, 0.0, 1.0));
    return max(environment_map.SampleLevel(environment_sampler, wrapped_uv, 0.0).rgb, float3(0.0))
        * max(environment_intensity, 0.0);
}

float3 sample_environment_blurred(float3 direction, float blur_radius)
{
    float2 uv = environment_equirect_uv(direction);
    float radius = clamp(blur_radius, 0.0, 0.12);
    if (radius <= 0.000001)
    {
        return sample_environment_uv(uv);
    }

    float2 u_offset = float2(radius, 0.0);
    float2 v_offset = float2(0.0, radius * 2.0);
    float3 color = sample_environment_uv(uv) * 0.20;
    color += (
        sample_environment_uv(uv + u_offset)
        + sample_environment_uv(uv - u_offset)
        + sample_environment_uv(uv + v_offset)
        + sample_environment_uv(uv - v_offset)
    ) * 0.12;
    color += (
        sample_environment_uv(uv + u_offset + v_offset)
        + sample_environment_uv(uv + u_offset - v_offset)
        + sample_environment_uv(uv - u_offset + v_offset)
        + sample_environment_uv(uv - u_offset - v_offset)
    ) * 0.08;
    return color;
}

#ifndef OMNISURG_TISSUE_CUSTOM_CLEARCOAT_REFLECTION
float3 tissue_clearcoat_reflection(
    float3 coat_reflection,
    float coat_roughness,
    float3 world_position,
    float3 coat_normal)
{
    return sample_environment_blurred(
        coat_reflection,
        lerp(0.006, 0.095, saturate(coat_roughness)));
}
#endif

#ifndef OMNISURG_TISSUE_CUSTOM_HEADLIGHT_VISIBILITY
float tissue_headlight_visibility(float3 world_position, float3 normal)
{
    return 1.0;
}
#endif

[shader("fragment")]
float4 OMNISURG_TISSUE_FRAGMENT_ENTRY(V2F input, bool is_front_face : SV_IsFrontFace) : SV_Target
{
    float3 view_dir = normalize(camera_pos - input.world_position);

    float face_sign = is_front_face ? 1.0 : -1.0;
    float3 geom_normal = normalize_or(input.world_normal, -camera_forward) * face_sign;
    float3 tangent = input.world_tangent.xyz;
    tangent = tangent - geom_normal * dot(geom_normal, tangent);
    tangent = normalize_or(tangent, fallback_tangent(geom_normal));
    float tangent_sign = input.world_tangent.w < 0.0 ? -1.0 : 1.0;
    float3 bitangent = normalize_or(
        cross(geom_normal, tangent) * tangent_sign,
        fallback_tangent(geom_normal));

    float heat_mask = heat_mask_tex.Sample(material_sampler, input.uv).r;
    float blood_mask = blood_mask_tex.Sample(material_sampler, input.uv).r;
    float damage_layer_mask = max(layer_mask_damage_tex.Sample(material_sampler, input.uv).r, heat_mask);
    float coag_layer_mask = max(layer_mask_coag_tex.Sample(material_sampler, input.uv).r, heat_mask);
    float blood_layer_mask = max(layer_mask_blood_tex.Sample(material_sampler, input.uv).r, blood_mask);
    float damage_blend = saturate(input.vertex_color.r * blend_damage * damage_layer_mask);
    float coag_blend = saturate(input.vertex_color.g * blend_coag * coag_layer_mask);
    float blood_blend = saturate(input.vertex_color.b * blend_blood * blood_layer_mask);

    float3 sampled_normal = blend_normal_layers(input.uv, damage_blend, coag_blend, blood_blend);
    sampled_normal.xy *= normal_strength;
    sampled_normal = normalize(sampled_normal);
    float3 normal = normalize(
        tangent * sampled_normal.x
        + bitangent * sampled_normal.y
        + geom_normal * sampled_normal.z);
    float normal_hemisphere = dot(normal, geom_normal);
    if (normal_hemisphere < 0.0001)
        normal = normalize_or(normal + geom_normal * (0.0001 - normal_hemisphere), geom_normal);

    float3 blended_diffuse = blend_rgb_layers(input.uv, damage_blend, coag_blend, blood_blend);
    float3 base_color = blended_diffuse * mesh_color.rgb * max(color_gain, 0.0);
    float spec_mask = blend_spec_layers(input.uv, damage_blend, coag_blend, blood_blend);

    KeyLightSample key_sample = compute_headlight_key_light(input.world_position);
    float3 key_light = key_sample.direction;
    float3 fill_light = normalize_or(light_fill_dir, float3(1.0, 0.0, 0.0));
    float3 back_light = normalize_or(light_back_dir, float3(0.0, 0.0, -1.0));
    float fill_scale = directional_fill_multiplier();
    float headlight_visibility = tissue_headlight_visibility(input.world_position, normal);
    float headlight_direct = key_sample.attenuation * headlight_visibility;
    float key_diffuse = saturate(dot(normal, key_light));
    float fill_diffuse = saturate(dot(normal, fill_light));
    float backscatter = pow(saturate(dot(-normal, back_light)), 2.0);

    float blood_film = saturate(max(blood_blend, blood_mask) * blood_wetness);
    float wet_film = saturate(wetness + blood_film);
    float base_roughness = saturate(roughness_bias + (1.0 - spec_mask) * 0.35);
    base_roughness = apply_specular_aa(base_roughness, normal);
    float3 base_half_dir = normalize_or(key_light + view_dir, normal);
    float base_n_dot_l = saturate(dot(normal, key_light));
    float base_n_dot_v = saturate(dot(normal, view_dir));
    float base_n_dot_h = saturate(dot(normal, base_half_dir));
    float base_v_dot_h = saturate(dot(view_dir, base_half_dir));
    float base_spec_level = max(specular_scale * (0.25 + spec_mask), 0.0);
    float3 base_f0 = float3(clamp(base_spec_level * 0.08, 0.0, 0.18));
    float base_distribution = ggx_distribution(base_n_dot_h, base_roughness);
    float base_geometry = smith_ggx_geometry(base_n_dot_l, base_n_dot_v, base_roughness);
    float3 base_fresnel = fresnel_schlick(base_v_dot_h, base_f0);
    float3 base_specular = base_distribution * base_geometry * base_fresnel
        / max(4.0 * base_n_dot_l * base_n_dot_v, 0.001);

    float3 coat_normal = normalize_or(geom_normal, normal);
    float coat_roughness = apply_specular_aa(clamp(wet_roughness, 0.02, 0.6), coat_normal);
    float3 coat_half_dir = normalize_or(key_light + view_dir, coat_normal);
    float coat_n_dot_l = saturate(dot(coat_normal, key_light));
    float coat_n_dot_v = saturate(dot(coat_normal, view_dir));
    float coat_n_dot_h = saturate(dot(coat_normal, coat_half_dir));
    float coat_v_dot_h = saturate(dot(view_dir, coat_half_dir));
    float coat_wet_boost = wet_film * max(wet_spec_scale, 0.0);
    float coat_intensity = saturate(coat_weight) * (1.0 + coat_wet_boost);
    float coat_distribution = ggx_distribution(coat_n_dot_h, coat_roughness);
    float coat_geometry = smith_ggx_geometry(coat_n_dot_l, coat_n_dot_v, coat_roughness);
    float coat_fresnel_direct = clearcoat_fresnel_omnisurface(
        coat_v_dot_h,
        coat_ior);
    float coat_direct = coat_distribution * coat_geometry * coat_fresnel_direct
        / max(4.0 * coat_n_dot_l * coat_n_dot_v, 0.001);
    float coat_fresnel_environment = clearcoat_fresnel_omnisurface(
        coat_n_dot_v,
        coat_ior);
    float base_layer_attenuation = 1.0 - saturate(coat_fresnel_environment * coat_intensity);
    float3 coat_reflection = reflect(-view_dir, coat_normal);
    float3 coat_environment = tissue_clearcoat_reflection(
        coat_reflection,
        coat_roughness,
        input.world_position,
        coat_normal);
    float3 wet_base_color = lerp(base_color, base_color * 0.72, wet_film * 0.35);

    if (debug_mode == DEBUG_VERTEX_BLEND_RGB)
        return float4(saturate(input.vertex_color.rgb), mesh_color.a);
    if (debug_mode == DEBUG_MASKED_LAYER_WEIGHTS)
        return float4(saturate(float3(damage_blend, coag_blend, blood_blend)), mesh_color.a);
    if (debug_mode == DEBUG_BLENDED_DIFFUSE)
        return float4(saturate(blended_diffuse), mesh_color.a);
    if (debug_mode == DEBUG_BLENDED_NORMAL)
        return float4(saturate(normal * 0.5 + 0.5), mesh_color.a);
    if (debug_mode == DEBUG_SPEC_ROUGHNESS)
        return float4(saturate(float3(spec_mask, base_roughness, wet_film)), mesh_color.a);
    if (debug_mode == DEBUG_HEAT_BLOOD_MASKS)
        return float4(saturate(float3(heat_mask, blood_mask, blood_layer_mask)), mesh_color.a);
    if (debug_mode == DEBUG_RESERVED_MATERIAL_PARAMS)
        return float4(
            saturate(
                float3(
                    wetness,
                    wet_spec_scale * wet_roughness,
                    blood_wetness)
                + subsurface_color * subsurface_strength),
            mesh_color.a);

    float ndv = saturate(dot(normal, view_dir));
    float rim = pow(1.0 - ndv, 2.0) * rim_strength;
    float3 subsurface_light = subsurface_color * backscatter * saturate(subsurface_strength);

    float3 environment_ambient = sample_environment_blurred(normal, 0.04);
    float3 ambient_color = has_environment_map != 0 && environment_lighting_enabled != 0
        ? environment_ambient * 0.55
        : light_ambient_color * ambient;
    float3 key_color = key_light_enabled != 0 ? light_key_color : float3(0.0);
    float3 fill_color = fill_light_enabled != 0 ? light_fill_color : float3(0.0);
    float3 rim_color = back_light_enabled != 0 ? light_rim_color : float3(0.0);
    float3 endoscope_color = endoscope_light_enabled != 0 ? endoscope_light_color : float3(0.0);
    float endoscope_diffuse = base_n_dot_v;

    float3 diffuse_light =
        ambient_color
        + key_color * key_diffuse * 0.88 * headlight_direct
        + fill_color * fill_diffuse * 0.42 * fill_scale
        + rim_color * backscatter * 0.16 * fill_scale
        + endoscope_color * endoscope_diffuse
        + subsurface_light;
    float3 color =
        wet_base_color * diffuse_light * base_layer_attenuation
        + key_color * base_specular * base_n_dot_l * base_layer_attenuation * headlight_direct
        + endoscope_color * base_specular * endoscope_diffuse * base_layer_attenuation
        + (
            key_color * coat_direct * coat_n_dot_l * headlight_direct
            + coat_environment * coat_fresnel_environment
        ) * coat_intensity
        + base_color * rim_color * rim * fill_scale;
    return float4(max(color, float3(0.0)), mesh_color.a);
}
