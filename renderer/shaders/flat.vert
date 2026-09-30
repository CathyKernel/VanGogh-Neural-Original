#version 300 es
// flat.vert - textured quad vertex shader ( GLSL ES 3.00 )
// Draws a full-painting quad: attribute positions are 0..1, scaled to
// painting pixels by uImageSize, then mapped to clip space.
// Used by the Flow Warp composite pass and the base-canvas pass.

precision highp float;

layout(location = 0) in vec2 aPos;    // 0..1 across the painting
layout(location = 1) in vec2 aUv;

out vec2 vUv;

uniform vec2  uImageSize;      // painting size ( px )
uniform vec2  uCameraCenter;   // camera center in painting px
uniform vec2  uScale;          // painting px -> clip space ( per axis )

void main() {
    vec2 world = aPos * uImageSize;
    gl_Position = vec4((world.x - uCameraCenter.x) * uScale.x,
                       -(world.y - uCameraCenter.y) * uScale.y, 0.0, 1.0);
    vUv = aUv;
}
