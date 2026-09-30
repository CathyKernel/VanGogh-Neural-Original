#version 300 es
// flat.frag - textured quad fragment shader ( GLSL ES 3.00 )
// Base canvas pass: the original painting, dimmed so instanced brush dabs
// read as impasto laid on top of it.

precision highp float;

#include "common.glsl"

in vec2 vUv;
out vec4 fragColor;

uniform sampler2D uTex;
uniform float uDim;      // 0..1 brightness scale
uniform float uTime;

void main() {
    vec3 c = texture(uTex, clamp(vUv, vec2(0.0), vec2(1.0))).rgb;
    fragColor = vec4(c * uDim + grain(vUv, uTime) * 0.02, 1.0);
}
