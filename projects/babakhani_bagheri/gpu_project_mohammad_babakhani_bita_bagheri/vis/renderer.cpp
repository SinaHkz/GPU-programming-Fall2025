#include "renderer.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {
void checkGlError(const char *file, int line, const char *label, bool enabled) {
  if (!enabled) return;
  GLenum err = glGetError();
  if (err != GL_NO_ERROR) {
    std::fprintf(stderr, "GL error 0x%x at %s:%d (%s)\n", err, file, line, label);
    while ((err = glGetError()) != GL_NO_ERROR) {
    }
  }
}

#define GL_CHECK(label, enabled) checkGlError(__FILE__, __LINE__, (label), (enabled))

const char *kFallbackParticleVert = R"glsl(
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
)glsl";

const char *kFallbackParticleFrag = R"glsl(
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
)glsl";

const char *kFallbackLineVert = R"glsl(
#version 330 core
layout(location = 0) in vec3 aPos;

uniform mat3 uRot;
uniform vec3 uPan;
uniform float uScale;

void main() {
  vec3 p = uRot * (aPos + uPan);
  gl_Position = vec4(p * uScale, 1.0);
}
)glsl";

const char *kFallbackLineFrag = R"glsl(
#version 330 core
uniform vec4 uColor;
out vec4 FragColor;
void main() {
  FragColor = uColor;
}
)glsl";

std::string loadTextFile(const std::string &path) {
  std::ifstream file(path);
  if (!file) {
    return {};
  }
  std::stringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

unsigned int compileShader(unsigned int type, const char *src) {
  unsigned int shader = glCreateShader(type);
  glShaderSource(shader, 1, &src, nullptr);
  glCompileShader(shader);
  int success = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    char log[512];
    glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
    std::fprintf(stderr, "Shader compile error: %s\n", log);
  }
  return shader;
}

unsigned int buildProgram(const char *vsSrc, const char *fsSrc) {
  unsigned int vs = compileShader(GL_VERTEX_SHADER, vsSrc);
  unsigned int fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);
  unsigned int program = glCreateProgram();
  glAttachShader(program, vs);
  glAttachShader(program, fs);
  glLinkProgram(program);

  int success = 0;
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if (!success) {
    char log[512];
    glGetProgramInfoLog(program, sizeof(log), nullptr, log);
    std::fprintf(stderr, "Program link error: %s\n", log);
  }

  glDeleteShader(vs);
  glDeleteShader(fs);
  return program;
}

void scrollCallback(GLFWwindow *window, double, double yoffset) {
  Renderer *renderer = static_cast<Renderer *>(glfwGetWindowUserPointer(window));
  if (renderer) {
    renderer->addScrollDelta(static_cast<float>(yoffset));
  }
}
} // namespace

bool Renderer::init(int width, int height, const char *title, int n,
                    const std::vector<float> &colors,
                    const std::vector<float> &sizes,
                    int trailLen, float viewRadius, float pointSizeScale) {
  n_ = n;
  trailLen_ = trailLen;
  viewRadius_ = viewRadius;
  pointSizeScale_ = pointSizeScale;

  if (!glfwInit()) {
    std::fprintf(stderr, "Failed to initialize GLFW\n");
    return false;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
  if (!window_) {
    std::fprintf(stderr, "Failed to create GLFW window\n");
    glfwTerminate();
    return false;
  }

  glfwMakeContextCurrent(window_);
  glfwSetWindowUserPointer(window_, this);
  glfwSwapInterval(1);
  glfwSetScrollCallback(window_, scrollCallback);

  yaw_ = 0.4f;
  pitch_ = -0.3f;

  if (glewInit() != GLEW_OK) {
    std::fprintf(stderr, "Failed to initialize GLEW\n");
    glfwDestroyWindow(window_);
    glfwTerminate();
    window_ = nullptr;
    return false;
  }

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_PROGRAM_POINT_SIZE);

  std::string particleVert = loadTextFile("shaders/particle.vert");
  std::string particleFrag = loadTextFile("shaders/particle.frag");
  std::string lineVert = loadTextFile("shaders/line.vert");
  std::string lineFrag = loadTextFile("shaders/line.frag");

  particleShader_ = buildProgram(
      particleVert.empty() ? kFallbackParticleVert : particleVert.c_str(),
      particleFrag.empty() ? kFallbackParticleFrag : particleFrag.c_str());
  lineShader_ = buildProgram(
      lineVert.empty() ? kFallbackLineVert : lineVert.c_str(),
      lineFrag.empty() ? kFallbackLineFrag : lineFrag.c_str());

  setupBuffers(colors, sizes);
  return true;
}

void Renderer::setupBuffers(const std::vector<float> &colors,
                            const std::vector<float> &sizes) {
  glGenVertexArrays(1, &vao_);
  glBindVertexArray(vao_);

  glGenBuffers(1, &vboPos_);
  glBindBuffer(GL_ARRAY_BUFFER, vboPos_);
  glBufferData(GL_ARRAY_BUFFER, n_ * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(0);

  glGenBuffers(1, &vboColor_);
  glBindBuffer(GL_ARRAY_BUFFER, vboColor_);
  glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(float), colors.data(), GL_DYNAMIC_DRAW);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(1);

  glGenBuffers(1, &vboSize_);
  glBindBuffer(GL_ARRAY_BUFFER, vboSize_);
  glBufferData(GL_ARRAY_BUFFER, sizes.size() * sizeof(float), sizes.data(), GL_DYNAMIC_DRAW);
  glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(2);

  glBindVertexArray(0);

  if (trailLen_ > 0) {
    glGenVertexArrays(1, &vaoTrail_);
    glBindVertexArray(vaoTrail_);
    glGenBuffers(1, &vboTrailPos_);
    glBindBuffer(GL_ARRAY_BUFFER, vboTrailPos_);
    glBufferData(GL_ARRAY_BUFFER, trailLen_ * n_ * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
  }

  glGenVertexArrays(1, &vaoVelocity_);
  glBindVertexArray(vaoVelocity_);
  glGenBuffers(1, &vboVelocity_);
  glBindBuffer(GL_ARRAY_BUFFER, vboVelocity_);
  glBufferData(GL_ARRAY_BUFFER, n_ * 6 * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(0);
  glBindVertexArray(0);

  glGenVertexArrays(1, &vaoClose_);
  glBindVertexArray(vaoClose_);
  glGenBuffers(1, &vboClose_);
  glBindBuffer(GL_ARRAY_BUFFER, vboClose_);
  glBufferData(GL_ARRAY_BUFFER, n_ * 2 * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(0);
  glBindVertexArray(0);

  glGenVertexArrays(1, &vaoAux_);
  glBindVertexArray(vaoAux_);
  glGenBuffers(1, &vboAux_);
  glBindBuffer(GL_ARRAY_BUFFER, vboAux_);
  glBufferData(GL_ARRAY_BUFFER, n_ * 2 * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(0);
  glBindVertexArray(0);
}

void Renderer::updatePositions(const std::vector<float> &positions) {
  glBindBuffer(GL_ARRAY_BUFFER, vboPos_);
  glBufferSubData(GL_ARRAY_BUFFER, 0, positions.size() * sizeof(float), positions.data());
  GL_CHECK("updatePositions", glDebug_);
}

void Renderer::updateColors(const std::vector<float> &colors) {
  glBindBuffer(GL_ARRAY_BUFFER, vboColor_);
  glBufferSubData(GL_ARRAY_BUFFER, 0, colors.size() * sizeof(float), colors.data());
  GL_CHECK("updateColors", glDebug_);
}

void Renderer::updateSizes(const std::vector<float> &sizes) {
  glBindBuffer(GL_ARRAY_BUFFER, vboSize_);
  glBufferSubData(GL_ARRAY_BUFFER, 0, sizes.size() * sizeof(float), sizes.data());
  GL_CHECK("updateSizes", glDebug_);
}

void Renderer::updateTrails(const std::vector<float> &trailPositions, int trailCount) {
  trailCount_ = trailCount;
  if (trailLen_ == 0) {
    return;
  }
  glBindBuffer(GL_ARRAY_BUFFER, vboTrailPos_);
  glBufferSubData(GL_ARRAY_BUFFER, 0, trailPositions.size() * sizeof(float), trailPositions.data());
  GL_CHECK("updateTrails", glDebug_);
}

void Renderer::updateVelocityLines(const std::vector<float> &linePositions) {
  velocityCount_ = static_cast<int>(linePositions.size() / 3);
  if (velocityCount_ == 0) {
    return;
  }
  glBindBuffer(GL_ARRAY_BUFFER, vboVelocity_);
  glBufferSubData(GL_ARRAY_BUFFER, 0, linePositions.size() * sizeof(float), linePositions.data());
  GL_CHECK("updateVelocityLines", glDebug_);
}

void Renderer::updateCloseLines(const std::vector<float> &linePositions) {
  closeCount_ = static_cast<int>(linePositions.size() / 3);
  if (closeCount_ == 0) {
    return;
  }
  glBindBuffer(GL_ARRAY_BUFFER, vboClose_);
  glBufferSubData(GL_ARRAY_BUFFER, 0, linePositions.size() * sizeof(float), linePositions.data());
  GL_CHECK("updateCloseLines", glDebug_);
}

void Renderer::updateAuxLines(const std::vector<float> &linePositions) {
  auxCount_ = static_cast<int>(linePositions.size() / 3);
  if (auxCount_ == 0) {
    return;
  }
  glBindBuffer(GL_ARRAY_BUFFER, vboAux_);
  glBufferSubData(GL_ARRAY_BUFFER, 0, linePositions.size() * sizeof(float), linePositions.data());
  GL_CHECK("updateAuxLines", glDebug_);
}

void Renderer::processInput() {
  actions_ = RenderActions{};

  if (!window_) {
    return;
  }

  if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
    glfwSetWindowShouldClose(window_, GLFW_TRUE);
  }

  bool tPressed = glfwGetKey(window_, GLFW_KEY_T) == GLFW_PRESS;
  if (tPressed && !toggleTrailsLatch_) {
    trailsEnabled_ = !trailsEnabled_;
  }
  toggleTrailsLatch_ = tPressed;

  bool pausePressed = glfwGetKey(window_, GLFW_KEY_SPACE) == GLFW_PRESS;
  if (pausePressed && !pauseLatch_) {
    actions_.pauseToggle = true;
  }
  pauseLatch_ = pausePressed;

  bool fPressed = glfwGetKey(window_, GLFW_KEY_F) == GLFW_PRESS;
  if (fPressed && !focusLatch_) {
    actions_.focusSelected = true;
  }
  focusLatch_ = fPressed;

  bool nPressed = glfwGetKey(window_, GLFW_KEY_N) == GLFW_PRESS;
  if (nPressed && !focusClosestLatch_) {
    actions_.focusClosest = true;
  }
  focusClosestLatch_ = nPressed;

  bool rPressed = glfwGetKey(window_, GLFW_KEY_R) == GLFW_PRESS;
  if (rPressed && !resetLatch_) {
    actions_.resetCamera = true;
  }
  resetLatch_ = rPressed;

  bool nextPressed = glfwGetKey(window_, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS;
  if (nextPressed && !selectNextLatch_) {
    actions_.selectDelta = 1;
  }
  selectNextLatch_ = nextPressed;

  bool prevPressed = glfwGetKey(window_, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS;
  if (prevPressed && !selectPrevLatch_) {
    actions_.selectDelta = -1;
  }
  selectPrevLatch_ = prevPressed;

  bool plusPressed = (glfwGetKey(window_, GLFW_KEY_EQUAL) == GLFW_PRESS) ||
                     (glfwGetKey(window_, GLFW_KEY_KP_ADD) == GLFW_PRESS);
  if (plusPressed && !plusLatch_) {
    actions_.timeScaleDelta = 1.0f;
  }
  plusLatch_ = plusPressed;

  bool minusPressed = (glfwGetKey(window_, GLFW_KEY_MINUS) == GLFW_PRESS) ||
                      (glfwGetKey(window_, GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS);
  if (minusPressed && !minusLatch_) {
    actions_.timeScaleDelta = -1.0f;
  }
  minusLatch_ = minusPressed;

  if (scrollDelta_ != 0.0f) {
    float zoomFactor = std::pow(1.1f, scrollDelta_);
    zoom_ *= zoomFactor;
    zoom_ = std::clamp(zoom_, minZoom_, maxZoom_);
    scrollDelta_ = 0.0f;
  }

  float panStep = 0.02f * viewRadius_ / zoom_;
  if (glfwGetKey(window_, GLFW_KEY_LEFT) == GLFW_PRESS) {
    panX_ += panStep;
  }
  if (glfwGetKey(window_, GLFW_KEY_RIGHT) == GLFW_PRESS) {
    panX_ -= panStep;
  }
  if (glfwGetKey(window_, GLFW_KEY_UP) == GLFW_PRESS) {
    panY_ -= panStep;
  }
  if (glfwGetKey(window_, GLFW_KEY_DOWN) == GLFW_PRESS) {
    panY_ += panStep;
  }

  double x = 0.0;
  double y = 0.0;
  glfwGetCursorPos(window_, &x, &y);
  int middle = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_MIDDLE);
  int right = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT);

  if (middle == GLFW_PRESS && !mouseMiddle_) {
    mouseMiddle_ = true;
    lastX_ = x;
    lastY_ = y;
  } else if (middle == GLFW_RELEASE) {
    mouseMiddle_ = false;
  }

  if (right == GLFW_PRESS && !mouseRight_) {
    mouseRight_ = true;
    lastX_ = x;
    lastY_ = y;
  } else if (right == GLFW_RELEASE) {
    mouseRight_ = false;
  }

  if (mouseMiddle_ || mouseRight_) {
    int width = 0;
    int height = 0;
    glfwGetWindowSize(window_, &width, &height);
    float dx = static_cast<float>(x - lastX_);
    float dy = static_cast<float>(y - lastY_);
    lastX_ = x;
    lastY_ = y;

    if (width > 0 && height > 0) {
      if (mouseMiddle_) {
        float scale = (viewRadius_ / zoom_) * 2.0f;
        panX_ += -dx / static_cast<float>(width) * scale;
        panY_ += dy / static_cast<float>(height) * scale;
      }
      if (mouseRight_) {
        yaw_ += dx * 0.005f;
        pitch_ += dy * 0.005f;
        if (pitch_ > 1.5f) pitch_ = 1.5f;
        if (pitch_ < -1.5f) pitch_ = -1.5f;
      }
    }
  }
}

void Renderer::updateUniforms(float alphaMul) {
  float scale = zoom_ / viewRadius_;
  float cy = std::cos(yaw_);
  float sy = std::sin(yaw_);
  float cx = std::cos(pitch_);
  float sx = std::sin(pitch_);
  float rot[9] = {
      cy, 0.0f, -sy,
      sy * sx, cx, cy * sx,
      sy * cx, -sx, cy * cx
  };

  glUseProgram(particleShader_);
  int locRot = glGetUniformLocation(particleShader_, "uRot");
  int locPan = glGetUniformLocation(particleShader_, "uPan");
  int locScale = glGetUniformLocation(particleShader_, "uScale");
  int locPoint = glGetUniformLocation(particleShader_, "uPointScale");
  int locBloom = glGetUniformLocation(particleShader_, "uBloomBoost");
  int locBloomThresh = glGetUniformLocation(particleShader_, "uBloomThreshold");
  int locImpostor = glGetUniformLocation(particleShader_, "uImpostorThreshold");
  int locPlanetStyle = glGetUniformLocation(particleShader_, "uPlanetStyle");
  glUniformMatrix3fv(locRot, 1, GL_FALSE, rot);
  glUniform3f(locPan, panX_, panY_, panZ_);
  glUniform1f(locScale, scale);
  glUniform1f(locPoint, pointSizeScale_);
  glUniform1f(locBloom, bloomBoost_);
  glUniform1f(locBloomThresh, bloomThreshold_);
  glUniform1f(locImpostor, impostorThreshold_);
  if (locPlanetStyle >= 0) {
    glUniform1i(locPlanetStyle, planetStyle_ ? 1 : 0);
  }

  glUseProgram(lineShader_);
  int locRotL = glGetUniformLocation(lineShader_, "uRot");
  int locPanL = glGetUniformLocation(lineShader_, "uPan");
  int locScaleL = glGetUniformLocation(lineShader_, "uScale");
  glUniformMatrix3fv(locRotL, 1, GL_FALSE, rot);
  glUniform3f(locPanL, panX_, panY_, panZ_);
  glUniform1f(locScaleL, scale);
  (void)alphaMul;
}

void Renderer::drawTrails() {
  if (!trailsEnabled_ || trailLen_ == 0 || trailCount_ < 2) {
    return;
  }
  glUseProgram(lineShader_);
  int locColor = glGetUniformLocation(lineShader_, "uColor");
  glUniform4f(locColor, 0.6f, 0.6f, 0.7f, trailAlpha_);
  glBindVertexArray(vaoTrail_);
  int count = std::min(trailCount_, trailLen_);
  for (int i = 0; i < n_; ++i) {
    if (i == cometTrailIndex_) {
      glLineWidth(cometTrailWidth_);
    } else {
      glLineWidth(1.0f);
    }
    glDrawArrays(GL_LINE_STRIP, i * trailLen_, count);
    GL_CHECK("drawTrails", glDebug_);
  }
  glLineWidth(1.0f);
}

void Renderer::drawLines(unsigned int vao, int count, float r, float g, float b, float a) {
  if (count == 0) {
    return;
  }
  glUseProgram(lineShader_);
  int locColor = glGetUniformLocation(lineShader_, "uColor");
  glUniform4f(locColor, r, g, b, a);
  glBindVertexArray(vao);
  glDrawArrays(GL_LINES, 0, count);
  GL_CHECK("drawLines", glDebug_);
}

void Renderer::updateInput() {
  processInput();
}

void Renderer::render() {
  glClearColor(0.02f, 0.02f, 0.04f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);

  updateUniforms(1.0f);

  drawTrails();
  drawLines(vaoClose_, closeCount_, 1.0f, 0.2f, 0.2f, closeLineAlpha_);
  drawLines(vaoAux_, auxCount_, 0.7f, 0.9f, 1.0f, auxLineAlpha_);
  drawLines(vaoVelocity_, velocityCount_, 0.2f, 1.0f, 0.4f, velocityLineAlpha_);

  glUseProgram(particleShader_);
  glBindVertexArray(vao_);
  glDrawArrays(GL_POINTS, 0, n_);
  GL_CHECK("drawPoints", glDebug_);

  glfwSwapBuffers(window_);
  GL_CHECK("swapBuffers", glDebug_);
}

void Renderer::pollEvents() {
  glfwPollEvents();
}

RenderActions Renderer::consumeActions() {
  RenderActions out = actions_;
  actions_ = RenderActions{};
  return out;
}

bool Renderer::shouldClose() const {
  return window_ && glfwWindowShouldClose(window_);
}

void Renderer::shutdown() {
  if (vboPos_) glDeleteBuffers(1, &vboPos_);
  if (vboColor_) glDeleteBuffers(1, &vboColor_);
  if (vboSize_) glDeleteBuffers(1, &vboSize_);
  if (vao_) glDeleteVertexArrays(1, &vao_);
  if (vboTrailPos_) glDeleteBuffers(1, &vboTrailPos_);
  if (vaoTrail_) glDeleteVertexArrays(1, &vaoTrail_);
  if (vboVelocity_) glDeleteBuffers(1, &vboVelocity_);
  if (vaoVelocity_) glDeleteVertexArrays(1, &vaoVelocity_);
  if (vboClose_) glDeleteBuffers(1, &vboClose_);
  if (vaoClose_) glDeleteVertexArrays(1, &vaoClose_);
  if (vboAux_) glDeleteBuffers(1, &vboAux_);
  if (vaoAux_) glDeleteVertexArrays(1, &vaoAux_);
  if (particleShader_) glDeleteProgram(particleShader_);
  if (lineShader_) glDeleteProgram(lineShader_);
  if (window_) {
    glfwDestroyWindow(window_);
    glfwTerminate();
    window_ = nullptr;
  }
}

bool Renderer::trailsEnabled() const {
  return trailsEnabled_;
}

void Renderer::setTrailsEnabled(bool enabled) {
  trailsEnabled_ = enabled;
}

void Renderer::setViewRadius(float radius) {
  viewRadius_ = radius;
}

void Renderer::setPointSizeScale(float scale) {
  pointSizeScale_ = scale;
}

void Renderer::setImpostorThreshold(float threshold) {
  impostorThreshold_ = std::max(0.0f, threshold);
}

void Renderer::setPlanetStyle(bool enabled) {
  planetStyle_ = enabled;
}

void Renderer::setGlDebug(bool enabled) {
  glDebug_ = enabled;
}

void Renderer::setZoomLimits(float minZoom, float maxZoom) {
  minZoom_ = std::max(minZoom, 0.001f);
  maxZoom_ = std::max(maxZoom, minZoom_ + 0.001f);
  zoom_ = std::clamp(zoom_, minZoom_, maxZoom_);
}

void Renderer::setAdditiveBlend(bool enabled) {
  additiveBlend_ = enabled;
  if (additiveBlend_) {
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
  } else {
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  }
}

void Renderer::setBloom(float boost, float threshold) {
  bloomBoost_ = boost;
  bloomThreshold_ = threshold;
}

void Renderer::setTrailAlpha(float alpha) {
  trailAlpha_ = alpha;
}

void Renderer::setVelocityLineAlpha(float alpha) {
  velocityLineAlpha_ = alpha;
}

void Renderer::setCloseLineAlpha(float alpha) {
  closeLineAlpha_ = alpha;
}

void Renderer::setAuxLineAlpha(float alpha) {
  auxLineAlpha_ = alpha;
}

void Renderer::setCamera(float panX, float panY, float panZ, float zoom, float viewRadius) {
  panX_ = panX;
  panY_ = panY;
  panZ_ = panZ;
  zoom_ = std::clamp(zoom, minZoom_, maxZoom_);
  viewRadius_ = viewRadius;
}

void Renderer::focusOn(float x, float y, float z, float radius) {
  panX_ = -x;
  panY_ = -y;
  panZ_ = -z;
  viewRadius_ = std::max(radius, 0.1f);
}

void Renderer::setCometTrail(int index, float width) {
  cometTrailIndex_ = index;
  cometTrailWidth_ = width;
}

void Renderer::getCamera(float &panX, float &panY, float &panZ, float &yaw, float &pitch, float &zoom, float &viewRadius) const {
  panX = panX_;
  panY = panY_;
  panZ = panZ_;
  yaw = yaw_;
  pitch = pitch_;
  zoom = zoom_;
  viewRadius = viewRadius_;
}

void Renderer::getWindowSize(int &w, int &h) const {
  if (!window_) {
    w = 0;
    h = 0;
    return;
  }
  glfwGetWindowSize(window_, &w, &h);
}

void Renderer::addScrollDelta(float delta) {
  scrollDelta_ += delta;
}
