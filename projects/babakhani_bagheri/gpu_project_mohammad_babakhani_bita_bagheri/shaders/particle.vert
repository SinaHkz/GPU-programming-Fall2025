#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;
layout(location = 2) in float aSize;

uniform mat3 uRot;
uniform vec3 uPan;
uniform float uScale;
uniform float uPointScale;

out vec4 vColor;
out float vSize;

void main() {
  vec3 p = uRot * (aPos + uPan);
  gl_Position = vec4(p * uScale, 1.0);
  gl_PointSize = aSize * uPointScale;
  vColor = aColor;
  vSize = aSize;
}
