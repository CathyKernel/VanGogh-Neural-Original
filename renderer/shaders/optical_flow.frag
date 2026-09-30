// optical_flow.frag - RAFT flow motion library ( GLSL ES 3.00 )
//
// Not a complete fragment shader on its own: it is textually included by
// depth_parallax.frag and brush.vert -style programs through the JS
// #include resolver in main.js.  It turns the static RAFT flow field into
// a *living* motion field: strokes and texels oscillate along their local
// flow vectors with layered sinusoids, so motion is continuous, periodic
// ( no drift ) and spatially coherent with the painted swirls.

vec2 flowOscillation(vec2 uv, float time, sampler2D flowTex, float flowScale) {
    vec2 f = decodeFlow(flowTex, uv, flowScale);
    // two beat frequencies + a spatial phase ramp: swirl cores breathe at
    // a different phase than their arms, producing the "living paint" feel.
    // The amplitude swing is wide ( 0.22 .. 1.12 ) so the swirl motion is
    // clearly visible rather than a shy shimmer.
    float w1 = sin(time * 1.25 + uv.x * 8.5 + uv.y * 6.0);
    float w2 = sin(time * 0.72 - uv.x * 4.0 + uv.y * 9.5 + 1.7);
    float w3 = sin(time * 2.2 + (uv.x + uv.y) * 14.0);
    float amp = 0.22 + 0.78 * (0.5 + 0.5 * w1 * w2) + 0.12 * w3;
    return f * amp;
}

// parallax offset for a given scene depth ( near = 1 ) in *pixels*:
// near layers track the parallax vector strongly, far layers barely move.
vec2 parallaxOffset(vec2 pan, float layerDepth) {
    return pan * (0.18 + 0.82 * layerDepth);
}
