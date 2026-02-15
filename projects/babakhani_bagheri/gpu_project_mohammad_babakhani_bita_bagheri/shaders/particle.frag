#version 330 core
in vec4 vColor;
in float vSize;
out vec4 FragColor;

uniform float uBloomBoost;
uniform float uBloomThreshold;
uniform float uImpostorThreshold;
uniform int uPlanetStyle;

void main() {
  vec2 uv = gl_PointCoord - vec2(0.5);
  float d = length(uv);
  float core = smoothstep(0.25, 0.0, d);
  float falloff = smoothstep(0.55, 0.0, d);
  float glow = smoothstep(0.65, 0.0, d);
  float boost = smoothstep(uBloomThreshold, uBloomThreshold * 2.0, vSize) * uBloomBoost;
  vec3 base = vColor.rgb;
  vec3 color = base * (0.2 + core * 1.2) + glow * 0.35 + boost;
  float alpha = falloff;

  if (vSize > uImpostorThreshold) {
    vec2 p = gl_PointCoord * 2.0 - vec2(1.0);
    float r2 = dot(p, p);
    float z = sqrt(max(0.0, 1.0 - r2));
    vec3 normal = normalize(vec3(p, z));
    vec3 lightDir = normalize(vec3(0.3, 0.4, 1.0));
    float diff = clamp(dot(normal, lightDir), 0.0, 1.0);
    float rim = smoothstep(0.9, 0.2, r2);
    vec3 bandColor = base;
    if (uPlanetStyle == 1) {
      if (base.r > 0.9 && base.g > 0.8) {
        float flicker = 0.5 + 0.5 * sin((p.x + p.y) * 12.0);
        bandColor = mix(base, vec3(1.0, 0.6, 0.2), flicker * 0.5);
      } else if (base.b > 0.75 && base.g > 0.45) {
        float n = 0.5 + 0.5 * sin((p.x * 6.0 + p.y * 8.0) * 3.14159);
        vec3 land = vec3(0.15, 0.55, 0.2);
        bandColor = mix(base, land, 0.25 + 0.35 * n);
      } else if (base.r > 0.8 && base.g < 0.5) {
        float n = 0.5 + 0.5 * sin((p.x * 5.0 - p.y * 7.0) * 3.14159);
        bandColor = mix(base, base * 0.65, 0.2 + 0.4 * n);
      } else if (base.r > 0.8 && base.g > 0.7 && base.b < 0.65) {
        if (base.g > 0.72) {
          float bands = 0.5 + 0.5 * sin((p.y * 6.0 + p.x * 2.0) * 3.14159);
          bandColor = mix(base, base * 0.7, bands);
        } else {
          float bands = 0.5 + 0.5 * sin(p.y * 7.0 * 3.14159);
          bandColor = mix(base, base * 0.75, bands);
          float ring = smoothstep(0.78, 0.8, r2) - smoothstep(0.9, 0.92, r2);
          bandColor = mix(bandColor, vec3(0.9, 0.85, 0.7), ring * 0.7);
        }
      } else if (base.g > 0.75 && base.b > 0.75) {
        float haze = 0.5 + 0.5 * sin(p.x * 4.0 * 3.14159);
        bandColor = mix(base, base * 0.8, 0.2 + 0.3 * haze);
      } else if (base.b > 0.75 && base.g < 0.4) {
        float bands = 0.5 + 0.5 * sin(p.y * 8.0 * 3.14159);
        bandColor = mix(base, base * 0.6, bands);
      } else if (base.r > 0.55 && base.g > 0.55 && base.b > 0.55) {
        float n = 0.5 + 0.5 * sin((p.x * 9.0 + p.y * 9.0) * 3.14159);
        bandColor = mix(base, base * 0.65, 0.2 + 0.3 * n);
      }
    } else {
      float bands = 0.5 + 0.5 * sin((p.y * 6.0 + base.r * 3.0) * 3.14159);
      bandColor = mix(base, base * 0.7, bands);
    }
    color = bandColor * (0.3 + 0.7 * diff) + rim * 0.15;
    alpha = smoothstep(1.0, 0.85, r2);
  }

  FragColor = vec4(color, vColor.a * alpha);
}
