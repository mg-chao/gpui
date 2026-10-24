cbuffer TextureParameters : register(b0) {
    float4 destination;
    float4 affine;
    float4 translation_viewport;
    float4 content_clip;
    float4 opacity_padding;
};
Texture2D<float4> image_texture : register(t0);
SamplerState image_sampler : register(s0);

struct TextureVertex {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

TextureVertex vertex_main(uint index : SV_VertexID) {
    TextureVertex output;
    output.uv = float2(index & 1, (index >> 1) & 1);
    float2 local = output.uv * destination.zw;
    float2 position = destination.xy + translation_viewport.xy
        + float2(affine.x * local.x + affine.z * local.y,
                 affine.y * local.x + affine.w * local.y);
    output.position = float4(position.x * 2 / translation_viewport.z - 1,
                             1 - position.y * 2 / translation_viewport.w, 0, 1);
    return output;
}

float4 fragment_main(TextureVertex input) : SV_Target {
    if (input.position.x < content_clip.x || input.position.y < content_clip.y
        || input.position.x >= content_clip.z || input.position.y >= content_clip.w) {
        discard;
    }
    return image_texture.Sample(image_sampler, input.uv) * opacity_padding.x;
}
