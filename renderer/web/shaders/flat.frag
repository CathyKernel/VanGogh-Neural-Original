#version 300 es
// flat.frag - base canvas pass ( GLSL ES 3.00 )
// The original painting, dimmed so instanced brush dabs read as impasto
// laid on top.  The base samples with a *per-pixel depth parallax* offset:
// when you grab the canvas, near pixels glide with the pointer while the
// far sky barely moves - the painting never translates as one flat image.

precision highp float;

#include "common.glsl"

in vec2 vUv;
out vec4 fragColor;

uniform sampler2D uTex;
uniform sampler2D uDepth;     // 8-bit relative depth ( near = 1 = white )
uniform float uDim;           // 0..1 brightness scale
uniform float uTime;
uniform vec2  uPan;           // parallax vector ( painting px )
uniform float uParallax;      // 0..2
uniform vec2  uImageSize;     // px

void main() {
    float d = texture(uDepth, vUv).r;
    vec2 off = uPan * uParallax * (0.18 + 0.82 * d);
    vec2 uv = clamp(vUv - off / uImageSize, vec2(0.0), vec2(1.0));
    vec3 c = texture(uTex, uv).rgb;
    fragColor = vec4(c * uDim + grain(vUv, uTime) * 0.02, 1.0);
}
