#version 300 es
// depth_parallax.frag - Flow Warp mode ( GLSL ES 3.00 )
//
// Layer-aware parallax composite: each semantic RGBA layer is sampled with
// its own offset - camera pan scaled by the layer's median depth plus the
// RAFT flow oscillation - then composited back-to-front in paint order.
// The original pixel colours are never altered: motion comes purely from
// sampling geometry, which is the "neural-preserving" contract of this demo.

precision highp float;

#include "common.glsl"
#include "optical_flow.frag"

in vec2 vUv;                 // uv in painting space ( v = 0 at top row )
out vec4 fragColor;

uniform sampler2D uOriginal;
uniform sampler2D uDepth;
uniform sampler2D uFlow;

uniform sampler2D uLayer0;
uniform sampler2D uLayer1;
uniform sampler2D uLayer2;
uniform sampler2D uLayer3;
uniform sampler2D uLayer4;
uniform sampler2D uLayer5;

uniform float uLayerDepth[6];
uniform int   uLayerCount;

uniform vec2  uPan;            // camera pan in pixels
uniform float uParallax;       // 0..2   parallax strength
uniform float uFlowStrength;   // 0..2   flow warp amplitude multiplier
uniform float uTime;           // seconds
uniform float uFlowScale;      // px, from manifest.flow.scale
uniform vec2  uImageSize;      // px

// composite one layer; sampler passed as a macro argument because GLSL ES
// preprocessors vary in ## token-pasting support ( ANGLE rejects it ).
// Sampling at vUv - off makes the layer content FOLLOW the parallax vector
// ( near layers glide with the grab, the far sky barely moves ).
#define COMPOSITE(idx, tex)                                               \
    if (idx < uLayerCount) {                                              \
        float d = uLayerDepth[idx];                                       \
        vec2 off = parallaxOffset(uPan * uParallax, d)                    \
                 + flowOscillation(vUv, uTime, uFlow, uFlowScale)          \
                     * uFlowStrength * (0.42 + 0.58 * d);                  \
        vec4 c = texture(tex, clamp(vUv - off / uImageSize,               \
                                    vec2(0.001), vec2(0.999)));            \
        acc = vec4(acc.rgb * (1.0 - c.a) + c.rgb * c.a,                   \
                   max(acc.a, c.a));                                       \
    }

void main() {
    // the original base gets a *per-pixel* depth parallax offset ( the
    // uDepth texture was previously bound but unused ) - near pixels glide
    // more than far ones, which is what kills the flat whole-image drag.
    float dp = texture(uDepth, vUv).r;
    vec2 baseOff = uPan * uParallax * (0.18 + 0.82 * dp);
    vec4 acc = texture(uOriginal, clamp(vUv - baseOff / uImageSize,
                                        vec2(0.0), vec2(1.0)));
    COMPOSITE(0, uLayer0)
    COMPOSITE(1, uLayer1)
    COMPOSITE(2, uLayer2)
    COMPOSITE(3, uLayer3)
    COMPOSITE(4, uLayer4)
    COMPOSITE(5, uLayer5)

    // gentle vignette + canvas grain keep the composite feeling photographic
    vec2 q = vUv - 0.5;
    float vig = 1.0 - 0.32 * dot(q, q) * 2.4;
    vec3 col = acc.rgb * clamp(vig, 0.55, 1.0);
    col += grain(vUv, uTime) * 0.035;

    fragColor = vec4(col, 1.0);
}
