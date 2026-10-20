// Screen-space fluid: sprite depth and thickness passes, a bilateral depth blur and a composite over the
// scene target. From omnisurg_fluid.slang (783 lines) with the anisotropic ellipsoids reduced to spheres,
// the thickness blur, the nine debug views, the legacy screen-fraction blur and the blurred environment
// reflection dropped. A dead pool particle (ACTIVE bit clear) is a degenerate quad, so the instanced draw
// covers the whole pool with no compaction and no count readback.
struct FullscreenV2F
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

struct FluidDepthV2F
{
    float4 position : SV_Position;
    float3 camera_point : TEXCOORD0;
    float3 center_camera : TEXCOORD1;
    nointerpolation float radius : TEXCOORD2;
};

struct FluidDepthOut
{
    float eye_depth : SV_Target0;
    float hardware_depth : SV_Depth;
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

uniform float3 light_key_dir;
uniform float3 light_fill_dir;
uniform float3 light_back_dir;
uniform int key_light_enabled;
uniform int fill_light_enabled;
uniform int back_light_enabled;
uniform int endoscope_light_enabled;
uniform float3 light_key_color;
uniform float3 light_fill_color;
uniform float3 light_ambient_color;
uniform float3 light_rim_color;
uniform float3 endoscope_light_color;
uniform float specular_scale;
uniform float ambient;
uniform float rim_strength;

Texture2D<float4> environment_map;
SamplerState environment_sampler;
uniform int has_environment_map;
uniform int environment_lighting_enabled;
uniform float environment_intensity;
uniform float environment_rotation;
uniform float environment_pitch;

uniform int fluid_count;
uniform float fluid_radius;
uniform float fluid_thickness_scale;
uniform float fluid_blur;              // world-space blur radius (m)
uniform float fluid_ior;
uniform float fluid_reflection_strength;
uniform float fluid_refraction_strength;
uniform float fluid_absorption_strength;
uniform float2 output_size;
uniform float3 fluid_color;

StructuredBuffer<float3> fluid_positions;
StructuredBuffer<int> fluid_flags;

Texture2D<float4> scene_color_tex;
Texture2D<float> scene_depth_tex;
Texture2D<float> fluid_depth_tex;
Texture2D<float4> fluid_thickness_tex;
SamplerState post_sampler;

static const int ACTIVE_FLAG = 1;

float3 normalize_or(float3 value, float3 fallback)
{
    float value_len = length(value);
    return value_len > 0.000001 ? value / value_len : fallback;
}

float hardware_depth_from_eye_depth(float eye_depth)
{
    float depth = camera_far / max(camera_far - camera_near, 0.000001)
        - (camera_near * camera_far) / (max(camera_far - camera_near, 0.000001) * max(eye_depth, 0.000001));
    return saturate(depth);
}

float eye_depth_from_hardware_depth(float hardware_depth)
{
    float denominator = camera_far - saturate(hardware_depth) * (camera_far - camera_near);
    return (camera_near * camera_far) / max(denominator, 0.000001);
}

float4 project_position(float3 position)
{
    float3 rel = position - camera_pos;
    float camera_x = dot(rel, camera_right);
    float camera_y = dot(rel, camera_up);
    float camera_z = max(dot(rel, camera_forward), camera_near);

    float clip_x = camera_x * camera_inv_tan_half_fovy / max(camera_aspect, 0.0001);
    float clip_y = camera_y * camera_inv_tan_half_fovy;
    float clip_z = camera_z * camera_far / max(camera_far - camera_near, 0.000001)
        - camera_near * camera_far / max(camera_far - camera_near, 0.000001);
    return float4(clip_x, clip_y, clip_z, camera_z);
}

float2 quad_corner(uint vertex_id)
{
    uint corner = vertex_id % 6;
    if (corner == 0 || corner == 3)
        return float2(-1.0, -1.0);
    if (corner == 1)
        return float2(1.0, -1.0);
    if (corner == 2 || corner == 4)
        return float2(1.0, 1.0);
    return float2(-1.0, 1.0);
}

[shader("vertex")]
FluidDepthV2F depth_vertex_main(uint vertex_id : SV_VertexID, uint instance_id : SV_InstanceID)
{
    FluidDepthV2F output;
    uint max_point_index = fluid_count > 0 ? uint(fluid_count - 1) : 0;
    uint point_index = min(instance_id, max_point_index);
    if ((fluid_flags[point_index] & ACTIVE_FLAG) == 0)
    {
        output.position = float4(0.0, 0.0, 0.0, 1.0);  // dead particle: every corner on one point, nothing rasterises
        output.camera_point = float3(0.0, 0.0, 1.0);
        output.center_camera = float3(0.0, 0.0, 1.0);
        output.radius = 0.0;
        return output;
    }
    float radius = max(fluid_radius, 0.0);
    float3 center = fluid_positions[point_index];
    float2 corner = quad_corner(vertex_id) * radius * 1.05;
    float3 world_position = center + camera_right * corner.x + camera_up * corner.y;
    float3 rel = center - camera_pos;
    float3 center_camera = float3(dot(rel, camera_right), dot(rel, camera_up), dot(rel, camera_forward));

    output.position = project_position(world_position);
    output.camera_point = center_camera + float3(corner.x, corner.y, 0.0);
    output.center_camera = center_camera;
    output.radius = radius;
    return output;
}

// Ray from the eye through the sprite pixel against the particle's sphere (omnisurg's ellipsoid with unit axes).
bool intersect_fluid_sphere(FluidDepthV2F input, out float ray_t_near, out float ray_t_far, out float3 ray_dir)
{
    ray_dir = normalize_or(input.camera_point, float3(0.0, 0.0, 1.0));
    ray_t_near = 0.0;
    ray_t_far = 0.0;
    if (input.radius <= 0.0 || ray_dir.z <= 0.0)
        return false;
    float b = -2.0 * dot(input.center_camera, ray_dir);
    float c = dot(input.center_camera, input.center_camera) - input.radius * input.radius;
    float discriminant = b * b - 4.0 * c;
    if (discriminant <= 0.0)
        return false;
    float root = sqrt(discriminant);
    ray_t_near = 0.5 * (-b - root);
    ray_t_far = 0.5 * (-b + root);
    return ray_t_far > 0.0;
}

[shader("fragment")]
FluidDepthOut depth_fragment_main(FluidDepthV2F input)
{
    FluidDepthOut output;
    float ray_t_near;
    float ray_t_far;
    float3 ray_dir;
    if (!intersect_fluid_sphere(input, ray_t_near, ray_t_far, ray_dir))
        discard;
    float ray_t = max(ray_t_near, 0.0);
    float eye_depth = ray_t * ray_dir.z;
    if (eye_depth <= camera_near)
        discard;
    output.eye_depth = eye_depth;
    output.hardware_depth = hardware_depth_from_eye_depth(eye_depth);
    return output;
}

[shader("fragment")]
float4 thickness_fragment_main(FluidDepthV2F input) : SV_Target
{
    float ray_t_near;
    float ray_t_far;
    float3 ray_dir;
    if (!intersect_fluid_sphere(input, ray_t_near, ray_t_far, ray_dir))
        discard;
    float ray_t = max(ray_t_near, 0.0);
    float eye_depth = ray_t * ray_dir.z;
    if (eye_depth <= camera_near)
        discard;
    float thickness = max(ray_t_far - ray_t, 0.0) * max(fluid_thickness_scale, 0.0);
    return float4(thickness, thickness * saturate(mesh_color.rgb));
}

[shader("vertex")]
FullscreenV2F fullscreen_vertex_main(uint vertex_id : SV_VertexID)
{
    FullscreenV2F output;
    output.uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    output.position = float4(output.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return output;
}

int2 fluid_output_size()
{
    return int2(max(int(output_size.x), 1), max(int(output_size.y), 1));
}

float fluid_depth_at_pixel(int2 pixel)
{
    int2 max_pixel = fluid_output_size() - int2(1, 1);
    return fluid_depth_tex.Load(int3(clamp(pixel, int2(0, 0), max_pixel), 0));
}

float4 fluid_thickness_color_at_pixel(int2 pixel)
{
    int2 max_pixel = fluid_output_size() - int2(1, 1);
    return fluid_thickness_tex.Load(int3(clamp(pixel, int2(0, 0), max_pixel), 0));
}

// omnisurg fluid_world_blur_radius_pixels: the blur radius is a world length projected at the centre depth.
float fluid_blur_radius_pixels(float center_depth)
{
    int2 dims = fluid_output_size();
    float projected_radius = max(fluid_blur, 1.0e-6) * camera_inv_tan_half_fovy * float(dims.y) / max(center_depth, 0.000001);
    return min(12.0, max(projected_radius, 1.0));
}

// omnisurg world_depth_fluid_smooth_depth: separable-looking bilateral blur, spatial Gaussian times a depth-range Gaussian.
[shader("fragment")]
float smooth_fragment_main(FullscreenV2F input) : SV_Target
{
    int2 pixel = int2(input.position.xy);
    float center_depth = fluid_depth_at_pixel(pixel);
    if (center_depth <= 0.0)
        return 0.0;
    if (fluid_blur <= 0.0)
        return center_depth;
    float radius = fluid_blur_radius_pixels(center_depth);
    int blur_radius = clamp(int(ceil(radius)), 1, 12);
    float radius_inv = 1.0 / max(radius, 1.0e-6);
    float fractional_taps = float(blur_radius) - radius;
    float depth_falloff = 5.5 / max(fluid_blur / 0.01, 0.25);
    float total_weight = 0.0;
    float total_depth = 0.0;
    for (int y = -blur_radius; y <= blur_radius; ++y)
    {
        for (int x = -blur_radius; x <= blur_radius; ++x)
        {
            float sample_depth = fluid_depth_at_pixel(pixel + int2(x, y));
            if (sample_depth <= 0.0)
                continue;
            float r1 = length(float2(float(x), float(y))) * radius_inv;
            float spatial_weight = exp(-(r1 * r1));
            float r2 = (sample_depth - center_depth) * depth_falloff;
            float range_weight = exp(-(r2 * r2));
            float boundary_weight = step(radius, max(abs(float(x)), abs(float(y))));
            float fractional_weight = 1.0 - boundary_weight * fractional_taps;
            float weight = spatial_weight * range_weight * fractional_weight;
            total_depth += sample_depth * weight;
            total_weight += weight;
        }
    }
    return total_weight > 0.0 ? total_depth / total_weight : center_depth;
}

float3 world_position_from_eye_depth(float2 uv, float eye_depth)
{
    float tan_half_fovy = 1.0 / max(camera_inv_tan_half_fovy, 0.000001);
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float camera_x = ndc.x * eye_depth * tan_half_fovy * camera_aspect;
    float camera_y = ndc.y * eye_depth * tan_half_fovy;
    return camera_pos + camera_right * camera_x + camera_up * camera_y + camera_forward * eye_depth;
}

// omnisurg reconstruct_fluid_normal: finite differences on the smoothed depth, taking the nearer neighbour on
// each axis so silhouettes stay sharp.
float3 reconstruct_fluid_normal(float2 uv, int2 pixel, float center_depth)
{
    int2 dims = fluid_output_size();
    float2 inv_size = float2(1.0 / float(dims.x), 1.0 / float(dims.y));
    int sample_step = clamp(int(max(fluid_blur_radius_pixels(center_depth) * 0.35 + 0.5, 1.0)), 1, 8);
    float left_depth = fluid_depth_at_pixel(pixel + int2(-sample_step, 0));
    float right_depth = fluid_depth_at_pixel(pixel + int2(sample_step, 0));
    float up_depth = fluid_depth_at_pixel(pixel + int2(0, -sample_step));
    float down_depth = fluid_depth_at_pixel(pixel + int2(0, sample_step));
    float left_delta = left_depth > 0.0 ? abs(left_depth - center_depth) : 1.0e20;
    float right_delta = right_depth > 0.0 ? abs(right_depth - center_depth) : 1.0e20;
    float up_delta = up_depth > 0.0 ? abs(up_depth - center_depth) : 1.0e20;
    float down_delta = down_depth > 0.0 ? abs(down_depth - center_depth) : 1.0e20;
    bool use_left = left_delta < right_delta;
    bool use_up = up_delta < down_delta;
    float x_depth = use_left ? left_depth : right_depth;
    float y_depth = use_up ? up_depth : down_depth;
    if (x_depth <= 0.0)
        x_depth = center_depth;
    if (y_depth <= 0.0)
        y_depth = center_depth;

    float2 sample_offset = inv_size * float(sample_step);
    float2 x_uv = use_left ? uv - float2(sample_offset.x, 0.0) : uv + float2(sample_offset.x, 0.0);
    float2 y_uv = use_up ? uv + float2(0.0, -sample_offset.y) : uv + float2(0.0, sample_offset.y);
    float3 center_pos = world_position_from_eye_depth(uv, center_depth);
    float3 x_pos = world_position_from_eye_depth(x_uv, x_depth);
    float3 y_pos = world_position_from_eye_depth(y_uv, y_depth);
    float3 tangent_x = use_left ? center_pos - x_pos : x_pos - center_pos;
    float3 tangent_y = use_up ? y_pos - center_pos : center_pos - y_pos;
    float3 normal = normalize_or(cross(tangent_x, tangent_y), -camera_forward);
    float3 view_dir = normalize_or(camera_pos - center_pos, -camera_forward);
    return dot(normal, view_dir) < 0.0 ? -normal : normal;
}

float3 fluid_lighting(float3 normal, float3 view_dir)
{
    float3 key_light = normalize_or(light_key_dir, -camera_forward);
    float3 fill_light = normalize_or(light_fill_dir, camera_right);
    float3 back_light = normalize_or(light_back_dir, camera_up);
    float key_diffuse = key_light_enabled != 0 ? saturate(dot(normal, key_light)) : 0.0;
    float fill_diffuse = fill_light_enabled != 0 ? saturate(dot(normal, fill_light)) : 0.0;
    float rim = back_light_enabled != 0 ? pow(saturate(1.0 - dot(normal, view_dir)), 2.0) * saturate(dot(normal, back_light) * 0.5 + 0.5) : 0.0;
    float endoscope = endoscope_light_enabled != 0 ? saturate(dot(normal, view_dir)) : 0.0;
    return light_ambient_color * max(ambient, 0.0)
        + light_key_color * key_diffuse
        + light_fill_color * fill_diffuse * 0.65
        + light_rim_color * rim * max(rim_strength, 0.0)
        + endoscope_light_color * endoscope;
}

float2 environment_equirect_uv(float3 direction)
{
    float3 dir = normalize_or(direction, float3(0.0, 1.0, 0.0));
    float pitch_sin = sin(environment_pitch);
    float pitch_cos = cos(environment_pitch);
    dir = normalize_or(float3(dir.x, dir.y * pitch_cos - dir.z * pitch_sin, dir.y * pitch_sin + dir.z * pitch_cos), dir);
    float u = atan2(dir.z, dir.x) * 0.159154943 + environment_rotation;
    float v = acos(clamp(dir.y, -1.0, 1.0)) * 0.318309886;
    return float2(frac(u), saturate(v));
}

float3 sample_environment(float3 direction)
{
    if (has_environment_map == 0 || environment_lighting_enabled == 0)
        return float3(0.0);
    float2 uv = environment_equirect_uv(direction);
    return max(environment_map.SampleLevel(environment_sampler, float2(frac(uv.x), saturate(uv.y)), 0.0).rgb, float3(0.0))
        * max(environment_intensity, 0.0);
}

float3 sample_refracted_scene(float2 uv, float fluid_eye_depth, float3 fallback_color)
{
    float2 sample_uv = saturate(uv);
    float scene_hardware_depth = scene_depth_tex.Sample(post_sampler, sample_uv);
    float scene_eye_depth = eye_depth_from_hardware_depth(scene_hardware_depth);
    if (scene_hardware_depth < 0.9999 && scene_eye_depth <= fluid_eye_depth - 0.0005)
        return fallback_color;
    return max(scene_color_tex.Sample(post_sampler, sample_uv).rgb, float3(0.0));
}

// omnisurg composite_fragment_main without the debug views and the smoothed thickness (the raw thickness feeds
// an exponential, so its blockiness does not show), with one environment sample for the reflection.
[shader("fragment")]
float4 composite_fragment_main(FullscreenV2F input) : SV_Target
{
    float2 uv = input.uv;
    int2 pixel = int2(input.position.xy);
    float3 scene_color = max(scene_color_tex.Sample(post_sampler, uv).rgb, float3(0.0));
    float fluid_eye_depth = fluid_depth_at_pixel(pixel);
    if (fluid_eye_depth <= 0.0)
        return float4(scene_color, 1.0);
    float4 thickness_color = fluid_thickness_color_at_pixel(pixel);
    float thickness = max(thickness_color.x, 0.0);
    float3 surface_color = thickness > 1.0e-8 ? saturate(max(thickness_color.yzw, float3(0.0)) / thickness) : fluid_color;

    float scene_hardware_depth = scene_depth_tex.Sample(post_sampler, uv);
    float scene_eye_depth = eye_depth_from_hardware_depth(scene_hardware_depth);
    if (scene_hardware_depth < 0.9999 && scene_eye_depth <= fluid_eye_depth - 0.0005)
        return float4(scene_color, 1.0);

    float3 world_pos = world_position_from_eye_depth(uv, fluid_eye_depth);
    float3 normal = reconstruct_fluid_normal(uv, pixel, fluid_eye_depth);
    float3 view_dir = normalize_or(camera_pos - world_pos, -camera_forward);
    float n_dot_v = saturate(dot(normal, view_dir));
    float fluid_ior_safe = max(fluid_ior, 0.0001);
    float eta = 1.0 / fluid_ior_safe;
    float f0_base = (fluid_ior_safe - 1.0) / (fluid_ior_safe + 1.0);
    float f0 = saturate(f0_base * f0_base);
    float fresnel = f0 + (1.0 - f0) * pow(1.0 - n_dot_v, 5.0);

    float optical_depth = thickness;
    float density = saturate(1.0 - exp(-optical_depth * max(fluid_absorption_strength, 0.0)));
    float refraction_strength = max(fluid_refraction_strength, 0.0) * saturate(abs(fluid_ior_safe - 1.0)) * (0.002 + density * 0.018);
    float3 refracted_dir = refract(-view_dir, normal, eta);
    refracted_dir = dot(refracted_dir, refracted_dir) > 1.0e-8 ? normalize(refracted_dir) : -view_dir;
    float3 bend_dir = refracted_dir + view_dir;
    float2 refraction_offset = float2(dot(bend_dir, camera_right), -dot(bend_dir, camera_up)) * refraction_strength * (1.0 - fresnel);
    float3 refracted = sample_refracted_scene(uv + refraction_offset, fluid_eye_depth, scene_color);
    float3 lighting = fluid_lighting(normal, view_dir);
    float3 half_dir = normalize_or(normalize_or(light_key_dir, -camera_forward) + view_dir, normal);
    float specular = pow(saturate(dot(normal, half_dir)), 96.0) * max(specular_scale, 0.0);

    float3 extinction_rgb = -log(max(surface_color, float3(0.015, 0.015, 0.015))) * max(fluid_absorption_strength, 0.0);
    float3 transmittance = exp(-optical_depth * extinction_rgb);
    float alpha = saturate(1.0 - dot(transmittance, float3(0.2126, 0.7152, 0.0722)));
    float3 liquid = surface_color * lighting * (0.08 + density * 0.42);
    float3 transmitted = refracted * transmittance + liquid * (0.25 + alpha * 0.45);
    float3 reflected = sample_environment(reflect(-view_dir, normal)) + lighting * 0.12 + float3(specular);
    float reflection_mix = saturate(fresnel * max(fluid_reflection_strength, 0.0));
    float3 color = lerp(transmitted, reflected, reflection_mix);
    return float4(max(color, float3(0.0)), 1.0);
}
