#version 300 es
// brush.vert - Brush Stroke mode vertex shader ( GLSL ES 3.00 )
//
// One instance per stroke *segment* ( record stride 12 floats, written by
// inference/painterly.py; grouping must match this attribute order ):
//     aSegA.xy   - segment midpoint in painting pixels
//     aSegA.zw   - unit direction along the stroke
//     aSegB.x    - segment length ( px, includes overlap )
//     aSegB.y    - brush radius ( px )
//     aSegB.z    - stroke opacity
//     aSegB.w    - semantic layer index ( paint order )
//     aSegC.rgb  - colour sampled from the original painting
//     aSegC.w    - random phase for desynchronised motion
//
// Motion: the segment center oscillates along the RAFT flow vector at its
// position ( the "living paint" ), scaled by stroke strength; layer depth
// drives the camera parallax exactly like the warp mode.  The dab also
// breathes ( width pulsation ) and wiggles ( tiny rotation ) so every stroke
// feels hand-painted rather than stamped.

precision highp float;

#include "common.glsl"

layout(location = 0) in vec2 aCorner;      // quad corner in (-1..1)
layout(location = 1) in vec4 aSegA;        // center.xy, dir.xy
layout(location = 2) in vec4 aSegB;        // len, radius, opacity, layer
layout(location = 3) in vec4 aSegC;        // color.rgb, phase

out vec2 vLocal;      // corner coords for the dab profile
out vec3 vColor;
out float vOpacity;
out float vStreak;    // per-stroke random for the fragment shader
out float vShimmer;   // advection pulse - syncs the impasto glint with motion

uniform sampler2D uFlow;

uniform vec2  uImageSize;      // painting size ( px )
uniform vec2  uCameraCenter;   // pan center in px
uniform vec2  uScale;          // painting px -> clip space ( per axis )
uniform vec2  uPan;            // parallax vector ( painting px )
uniform float uParallax;
uniform float uStrokeMotion;   // 0..2 flow advection amplitude
uniform float uTime;
uniform float uTimeScale;
uniform float uFlowScale;
uniform float uBrushScale;
uniform float uLayerDepth[6];

vec2 toClip(vec2 p) {
    return vec2((p.x - uCameraCenter.x) * uScale.x,
                -(p.y - uCameraCenter.y) * uScale.y);
}

void main() {
    vec2 center = aSegA.xy;
    vec2 dir = aSegA.zw;
    float len = aSegB.x;
    float radius = aSegB.y;
    float opacity = aSegB.z;
    float layer = aSegB.w;
    vec3 color = aSegC.rgb;
    float phase = aSegC.w;

    // ---- flow advection ( periodic, drift free, amplified ) ----
    vec2 uv = center / uImageSize;
    vec2 flow = decodeFlow(uFlow, uv, uFlowScale);
    float beat = sin(uTime * uTimeScale * 0.9 + phase * 3.1)
               * sin(uTime * uTimeScale * 0.53 + uv.x * 7.0 + uv.y * 5.0 + phase);
    float pulse = 0.30 + 0.70 * beat;          // wide swing, strong "living" feel
    vec2 advect = flow * uStrokeMotion * pulse * 1.05;

    // ---- depth parallax for the segment's layer ----
    // uPan is a pure parallax vector: near layers ( depth ~1 ) follow it
    // fully, the far sky barely - no rigid whole-image translation.
    float depth = clamp(uLayerDepth[int(min(layer, 5.0))], 0.0, 1.0);
    vec2 world = center + advect + uPan * uParallax * (0.18 + 0.82 * depth);

    // ---- oriented dab quad: breathes and wiggles like live paint ----
    float breath = 1.0 + 0.16 * sin(uTime * uTimeScale * 1.7 + phase * 2.0);
    float wob = 0.05 * sin(uTime * uTimeScale * 0.8 + phase * 2.7);
    vec2 dirW = vec2(dir.x * cos(wob) - dir.y * sin(wob),
                     dir.x * sin(wob) + dir.y * cos(wob));
    vec2 axis = dirW * (len * 0.5 * aCorner.x)
              + vec2(-dirW.y, dirW.x) * (radius * uBrushScale * breath * aCorner.y);
    vec2 p = world + axis;

    gl_Position = vec4(toClip(p), 0.0, 1.0);
    vLocal = aCorner;
    vColor = color;
    vOpacity = opacity;
    vStreak = phase;
    vShimmer = pulse;
}
