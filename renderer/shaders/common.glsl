// common.glsl - shared helpers for both render modes ( GLSL ES 3.00 )
//
// The flow codec here MUST mirror inference/flow_codec.py:
//     R = (u / scale + 1) * 127.5     -> u = (R/255*2 - 1) * scale
//     G = (v / scale + 1) * 127.5     -> v = (G/255*2 - 1) * scale

vec2 decodeFlow(sampler2D flowTex, vec2 uv, float scale) {
    vec3 enc = texture(flowTex, uv).rgb;
    return (vec2(enc.r, enc.g) * 2.0 - 1.0) * scale;
}

float hash11(float p) {
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

float hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// canvas grain ( subtle, applied in both modes )
float grain(vec2 uv, float t) {
    return vnoise(uv * vec2(920.0, 700.0) + vec2(t * 13.0)) - 0.5;
}
