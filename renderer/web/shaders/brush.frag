#version 300 es
// brush.frag - Brush Stroke mode fragment shader ( GLSL ES 3.00 )
//
// Impasto dab profile:
//   * an almond body that tapers toward the stroke ends - the brush
//     lifting off the canvas instead of stopping dead;
//   * organic bristle bands across the stroke plus fine dry-brush
//     streaks, seeded per stroke so no two dabs are identical;
//   * a warm impasto ridge slightly above the spine that glints in sync
//     with the stroke's motion, and a softly darkened lower edge that
//     reads as paint thickness.
// Colour arrives from the vertex shader: it was sampled from the original
// painting during stroke extraction, so palette fidelity is preserved.

precision highp float;

#include "common.glsl"

in vec2 vLocal;      // (-1..1) quad coordinates, x along the stroke
in vec3 vColor;
in float vOpacity;
in float vStreak;    // per-stroke random phase
in float vShimmer;   // advection pulse ( from the vertex shader )

out vec4 fragColor;

uniform float uGlobalAlpha;   // fade-in / mode transition

void main() {
    float ax = clamp(vLocal.x, -1.0, 1.0);

    // almond taper: full width mid-stroke, 72% at the tips
    float taper = 0.72 + 0.28 * sqrt(max(0.0, 1.0 - ax * ax));
    float y = vLocal.y / taper;

    // soft body with a firm paint core
    float d = length(vec2(vLocal.x * 0.94, y));
    float body = 1.0 - smoothstep(0.18, 1.0, d);

    // organic bristle bands + fine dry-brush streaks, seeded per stroke
    float band = vnoise(vec2(vLocal.x * 5.5 + vStreak * 19.0,
                             vLocal.y * 1.6 + vStreak * 3.0));
    float fine = 0.5 + 0.5 * sin(vLocal.x * 38.0 + vStreak * 31.0 + band * 4.0);
    float bristle = mix(band, fine, 0.42);

    // impasto ridge: light catches the paint just above the spine
    float rt = (vLocal.y - 0.22 * taper) / 0.42;
    float ridge = exp(-rt * rt) * (0.40 + 0.60 * taper);
    // darker lower edge reads as paint thickness / self-shadowing
    float edgeShade = 1.0 - 0.14 * smoothstep(0.45, 1.0, abs(y));

    float alpha = vOpacity * body * (0.62 + 0.38 * bristle) * uGlobalAlpha;
    vec3 col = vColor * edgeShade * (0.92 + 0.16 * bristle);
    col += vec3(0.10, 0.088, 0.062) * ridge * (0.55 + 0.45 * vShimmer);
    col *= 0.97 + 0.06 * vShimmer;
    fragColor = vec4(col, alpha);
}
