#version 330 core
layout(location = 0) in vec3 aPos;

uniform mat3 uRot;
uniform vec3 uPan;
uniform float uScale;

void main() {
  vec3 p = uRot * (aPos + uPan);
  gl_Position = vec4(p * uScale, 1.0);
}
