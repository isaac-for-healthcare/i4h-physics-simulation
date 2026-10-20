// Flat-coloured PBR mesh shading (instruments, rod tubes, the sphere overlay). From omnisurg_mesh.slang
// (536 lines): the common uniform block, projection, headlight, environment sampling, shade_pbr_surface and the
// textured_vertex_main / pbr_fragment_main entry points; the texture sampling, tangent frame, uv stream and the flat,
// colored, point and plain textured variants are dropped (every draw bound the default white / flat-normal /
// half-grey maps, so the normal is the geometric normal and the specular level 0.5).
struct TexturedV2F
{
    float4 position : SV_Position;
    float3 world_position : WORLD_POSITION;
    float3 world_normal : NORMAL;
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
uniform float specular_scale;
uniform float ambient;
uniform float rim_strength;
Texture2D<float4> environment_map;
SamplerState environment_sampler;
uniform float material_roughness;
uniform float material_metallic;
uniform int has_environment_map;
uniform int environment_lighting_enabled;
uniform float environment_intensity;
uniform float environment_rotation;
uniform float environment_pitch;

float3 normalize_or(float3 value, float3 fallback)
{
    float value_len = length(value);
    return value_len > 0.000001 ? value / value_len : fallback;
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

float4 project_position(float3 position)
{
    return mul(camera_view_projection, float4(position, 1.0));
}

[shader("vertex")]
TexturedV2F textured_vertex_main(float3 position : POSITION, float3 normal : NORMAL)
{
    TexturedV2F output;
    output.position = project_position(position);
    output.world_position = position;
    output.world_normal = normal;
    return output;
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

float3 shade_pbr_surface(
    float3 world_position,
    float3 normal,
    float3 base_color,
    float specular_level,
    float roughness,
    float metallic)
{
    float3 view_dir = normalize(camera_pos - world_position);
    normal = normalize_or(normal, view_dir);
    KeyLightSample key_sample = compute_headlight_key_light(world_position);
    float3 key_light = key_sample.direction;
    float3 fill_light = normalize(light_fill_dir);
    float3 back_light = normalize(light_back_dir);
    float3 half_dir = normalize(key_light + view_dir);
    float fill_scale = directional_fill_multiplier();

    roughness = saturate(roughness);
    metallic = saturate(metallic);
    specular_level = saturate(specular_level);

    float key_diffuse = saturate(dot(normal, key_light));
    float fill_diffuse = saturate(dot(normal, fill_light));
    float back_wrap = pow(saturate(dot(-normal, back_light)), 2.0);
    float ndv = saturate(dot(normal, view_dir));
    float ndh = saturate(dot(normal, half_dir));
    float shininess = lerp(96.0, 22.0, roughness);
    float spec = pow(ndh, shininess)
        * specular_scale
        * lerp(0.20, 1.25, specular_level)
        * lerp(1.15, 0.35, roughness);
    float3 diffuse = base_color * (1.0 - metallic * 0.65);
    float3 specular_color = lerp(float3(1.0), base_color, metallic);

    float3 environment_ambient = sample_environment_blurred(normal, 0.04);
    float3 ambient_color = has_environment_map != 0 && environment_lighting_enabled != 0
        ? environment_ambient * 0.55
        : light_ambient_color * ambient;
    float3 key_color = key_light_enabled != 0 ? light_key_color : float3(0.0);
    float3 fill_color = fill_light_enabled != 0 ? light_fill_color : float3(0.0);
    float3 rim_color = back_light_enabled != 0 ? light_rim_color : float3(0.0);
    float3 endoscope_color = endoscope_light_enabled != 0 ? endoscope_light_color : float3(0.0);

    float rim = pow(1.0 - ndv, 2.0) * rim_strength;
    float endoscope_diffuse = ndv;
    float3 diffuse_light =
        ambient_color
        + key_color * key_diffuse * 0.95 * key_sample.attenuation
        + fill_color * fill_diffuse * 0.48 * fill_scale
        + rim_color * back_wrap * 0.20 * fill_scale
        + endoscope_color * endoscope_diffuse;
    return diffuse * diffuse_light
        + key_color * specular_color * spec * key_sample.attenuation
        + endoscope_color * specular_color * spec * endoscope_diffuse
        + diffuse * rim_color * rim * fill_scale;
}

[shader("fragment")]
float4 pbr_fragment_main(TexturedV2F input) : SV_Target
{
    float3 view_dir = normalize(camera_pos - input.world_position);
    float3 normal = normalize_or(input.world_normal, view_dir);
    normal = dot(normal, view_dir) < 0.0 ? -normal : normal;
    const float specular_level = 0.5;
    float roughness = saturate(material_roughness + (1.0 - specular_level) * 0.25);
    float3 color = shade_pbr_surface(input.world_position, normal, mesh_color.rgb, specular_level, roughness, material_metallic);
    return float4(max(color, float3(0.0)), mesh_color.a);
}
