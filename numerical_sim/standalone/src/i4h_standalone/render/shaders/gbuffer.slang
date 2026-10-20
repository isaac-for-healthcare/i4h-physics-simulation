// G-buffer for dataset capture: camera-space normal + eye depth (SV_Target0) and a per-object label (SV_Target1).
// Port of the canonical omnisurg/rendering/slang_shaders/omnisurg_recording.slang:1-76 (dataset_vertex, smooth_vertex,
// dataset_fragment). Edits: the camera uniforms are declared here instead of `#include "hex_flat.slang"`; the flat and
// instanced-sphere vertex entries and clamp_near_depth are dropped (every stream we draw carries normals and no
// spheres are recorded); the projection is the tissue/mesh one (mesh.slang project_position), so the labels line up
// with the RGB draw pixel for pixel.
uniform float3 camera_pos;
uniform float3 camera_right;
uniform float3 camera_up;
uniform float3 camera_forward;
uniform float camera_inv_tan_half_fovy;
uniform float camera_aspect;
uniform float camera_near;
uniform float camera_far;
uniform uint object_id;

struct DatasetVertex
{
    float4 position : SV_Position;
    float3 world_position : TEXCOORD0;
    float3 normal : TEXCOORD1;
};

struct DatasetOutput
{
    float4 normal_depth : SV_Target0;
    uint label : SV_Target1;
};

float3 normalize_or(float3 v, float3 fallback)
{
    float len = length(v);
    return len > 1.0e-12 ? v / len : fallback;
}

[shader("vertex")]
DatasetVertex smooth_vertex(float3 position : POSITION, float3 normal : NORMAL)
{
    DatasetVertex output;
    float3 rel = position - camera_pos;
    float camera_x = dot(rel, camera_right);
    float camera_y = dot(rel, camera_up);
    float camera_z = dot(rel, camera_forward);
    float clip_x = camera_x * camera_inv_tan_half_fovy / max(camera_aspect, 0.0001);
    float clip_y = camera_y * camera_inv_tan_half_fovy;
    float clip_z = camera_z * camera_far / (camera_far - camera_near)
        - camera_near * camera_far / (camera_far - camera_near);
    output.position = float4(clip_x, clip_y, clip_z, camera_z);
    output.world_position = position;
    output.normal = normal;
    return output;
}

[shader("fragment")]
DatasetOutput dataset_fragment(DatasetVertex input)
{
    float3 normal = input.normal;
    if (dot(normal, normal) < 1.0e-12)
        normal = cross(ddx(input.world_position), ddy(input.world_position));
    normal = normalize_or(normal, -camera_forward);
    if (dot(normal, camera_pos - input.world_position) < 0)
        normal = -normal;
    DatasetOutput output;
    // Camera coordinates: right, up, forward. Face the viewing camera.
    output.normal_depth = float4(dot(normal, camera_right), dot(normal, camera_up),
        dot(normal, camera_forward), dot(input.world_position - camera_pos, camera_forward));
    output.label = object_id;
    return output;
}
