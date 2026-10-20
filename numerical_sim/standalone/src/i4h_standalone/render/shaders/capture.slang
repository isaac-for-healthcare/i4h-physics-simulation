// Dataset capture: pack the display frame, the G-buffer's depth, labels and normals into one uint3 per pixel
// (rgb8 | depth16 + label16 | normal rgb8: 12 bytes) for the host to unpack into video planes.
// Port of the canonical omnisurg/rendering/slang_shaders/omnisurg_recording.slang:78-116 (pack_frame). Edits: the frame
// size is a uniform (omnisurg fixed 1024x576); the depth comes from the G-buffer's eye depth (nd.w) as there; the
// label texture is 32-bit and masked to 16 bits; the frame's offset into the buffer is 0 (one staging buffer per
// frame). The lens distortion and the scope mask are re-applied so depth, labels and normals stay pixel-aligned with the
// post-processed rgb.
Texture2D<float4> capture_rgb;
Texture2D<float4> capture_normal_depth;
Texture2D<uint> capture_labels;
RWStructuredBuffer<uint3> capture_frames;
uniform uint width;
uniform uint height;
uniform float depth_max;
uniform float camera_aspect;
uniform int dataset_distortion_enabled;
uniform float dataset_distortion_strength;
uniform float dataset_distortion_zoom;
uniform int dataset_scope_enabled;
uniform float dataset_scope_radius;

[shader("compute")]
[numthreads(16, 16, 1)]
void pack_frame(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= width || tid.y >= height)
        return;
    float2 size = float2(float(width), float(height));
    uint3 rgb = uint3(round(saturate(capture_rgb[tid.xy].rgb) * 255.0));
    float2 uv = (float2(tid.xy) + 0.5) / size;
    float2 centered = (uv * 2.0 - 1.0) * float2(camera_aspect, 1);
    float radius = length(centered);
    if (dataset_distortion_enabled != 0)
    {
        centered *= dataset_distortion_zoom * (1.0 + dataset_distortion_strength * dot(centered, centered));
        uv = centered / float2(camera_aspect, 1) * 0.5 + 0.5;
    }
    int2 pixel = clamp(int2(uv * size), int2(0), int2(int(width) - 1, int(height) - 1));
    float4 nd = capture_normal_depth[pixel];
    uint label = capture_labels[pixel] & 0xFFFF;
    if (dataset_scope_enabled != 0 && radius >= dataset_scope_radius)
        label = 0;
    // Fixed-range depth; never normalised per frame.
    uint depth = label == 0 ? 0 : uint(round(saturate(1.0 - nd.w / depth_max) * 65535.0));
    uint3 normal = label == 0 ? uint3(0) : uint3(round(saturate(nd.xyz * 0.5 + 0.5) * 255.0));
    uint index = tid.y * width + tid.x;
    capture_frames[index] = uint3(rgb.x | (rgb.y << 8) | (rgb.z << 16) | 0xff000000,
        depth | (label << 16), normal.x | (normal.y << 8) | (normal.z << 16) | 0xff000000);
}
