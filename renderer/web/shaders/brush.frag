#version 300 es
// brush.frag - Brush Stroke mode fragment shader ( GLSL ES 3.00 )
//
// A single dab profile: an elongated soft body with streaky bristle noise
// along the stroke axis - a cheap stand-in for an impasto brush texture.
// Colour arrives from the vertex shader: it was sampled from the original
// painting during stroke extraction, so palette fidelity is preserved.

precision highp float;

#include "common.glsl"

in vec2 vLocal;      // (-1..1) quad coordinates, x along the stroke
in vec3 vColor;
in float vOpacity;
in float vStreak;

out vec4 fragColor;

uniform float uGlobalAlpha;   // fade-in / mode transition

void main() {
    // soft elongated body
    float d = length(vLocal * vec2(0.92, 1.06));
    float body = smoothstep(1.0, 0.32, d);

    // bristle streaks: 1D noise along the stroke axis, seeded per stroke
    float streakPos = vLocal.x * 6.0 + vStreak * 13.0;
    float streak = 0.5 + 0.5 * sin(streakPos * 3.1 + sin(streakPos * 1.7 + vStreak) * 2.3);
    float fine = 0.5 + 0.5 * sin(vLocal.x * 43.0 + vStreak * 31.0);
    streak = mix(streak, fine, 0.35);

    float alpha = vOpacity * body * (0.58 + 0.42 * streak) * uGlobalAlpha;
    vec3 col = vColor * (0.90 + 0.18 * streak);       // slight impasto shimmer
    col += vec3(0.05) * smoothstep(0.6, 1.0, body);   // thin highlight ridge
    fragColor = vec4(col, alpha);
}
