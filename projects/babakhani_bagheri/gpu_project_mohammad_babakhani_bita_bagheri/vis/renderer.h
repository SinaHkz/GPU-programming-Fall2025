#pragma once

#include <vector>

struct GLFWwindow;

struct RenderActions {
  float timeScaleDelta = 0.0f;
  bool pauseToggle = false;
  bool focusSelected = false;
  bool focusClosest = false;
  bool resetCamera = false;
  int selectDelta = 0;
};

class Renderer {
public:
  bool init(int width, int height, const char *title, int n,
            const std::vector<float> &colors,
            const std::vector<float> &sizes,
            int trailLen, float viewRadius, float pointSizeScale);

  void updatePositions(const std::vector<float> &positions);
  void updateColors(const std::vector<float> &colors);
  void updateSizes(const std::vector<float> &sizes);
  void updateTrails(const std::vector<float> &trailPositions, int trailCount);
  void updateVelocityLines(const std::vector<float> &linePositions);
  void updateCloseLines(const std::vector<float> &linePositions);
  void updateAuxLines(const std::vector<float> &linePositions);

  void updateInput();
  void render();
  void pollEvents();
  RenderActions consumeActions();
  bool shouldClose() const;
  void shutdown();

  bool trailsEnabled() const;
  void setTrailsEnabled(bool enabled);
  void setViewRadius(float radius);
  void setPointSizeScale(float scale);
  void setZoomLimits(float minZoom, float maxZoom);
  void setImpostorThreshold(float threshold);
  void setPlanetStyle(bool enabled);
  void setGlDebug(bool enabled);
  void setAdditiveBlend(bool enabled);
  void setBloom(float boost, float threshold);
  void setTrailAlpha(float alpha);
  void setVelocityLineAlpha(float alpha);
  void setCloseLineAlpha(float alpha);
  void setAuxLineAlpha(float alpha);
  void setCamera(float panX, float panY, float panZ, float zoom, float viewRadius);
  void focusOn(float x, float y, float z, float radius);
  void setCometTrail(int index, float width);
  void getCamera(float &panX, float &panY, float &panZ, float &yaw, float &pitch, float &zoom, float &viewRadius) const;
  void getWindowSize(int &w, int &h) const;
  void addScrollDelta(float delta);

private:
  void processInput();
  void updateUniforms(float alphaMul);
  void setupBuffers(const std::vector<float> &colors,
                    const std::vector<float> &sizes);
  void drawTrails();
  void drawLines(unsigned int vao, int count, float r, float g, float b, float a);

  GLFWwindow *window_ = nullptr;
  unsigned int particleShader_ = 0;
  unsigned int lineShader_ = 0;
  unsigned int vao_ = 0;
  unsigned int vboPos_ = 0;
  unsigned int vboColor_ = 0;
  unsigned int vboSize_ = 0;

  unsigned int vaoTrail_ = 0;
  unsigned int vboTrailPos_ = 0;

  unsigned int vaoVelocity_ = 0;
  unsigned int vboVelocity_ = 0;

  unsigned int vaoClose_ = 0;
  unsigned int vboClose_ = 0;
  unsigned int vaoAux_ = 0;
  unsigned int vboAux_ = 0;

  int n_ = 0;
  int trailLen_ = 0;
  int trailCount_ = 0;
  int velocityCount_ = 0;
  int closeCount_ = 0;
  int auxCount_ = 0;

  float viewRadius_ = 10.0f;
  float zoom_ = 1.0f;
  float minZoom_ = 0.1f;
  float maxZoom_ = 100.0f;
  float panX_ = 0.0f;
  float panY_ = 0.0f;
  float panZ_ = 0.0f;
  float yaw_ = 0.0f;
  float pitch_ = 0.0f;
  float pointSizeScale_ = 4.0f;
  float impostorThreshold_ = 0.1f;
  float bloomBoost_ = 0.0f;
  float bloomThreshold_ = 1.0f;
  float trailAlpha_ = 0.35f;
  float velocityLineAlpha_ = 0.7f;
  float closeLineAlpha_ = 0.9f;
  float auxLineAlpha_ = 0.7f;
  bool planetStyle_ = true;
  bool additiveBlend_ = false;
  bool glDebug_ = false;
  int cometTrailIndex_ = -1;
  float cometTrailWidth_ = 1.0f;

  bool trailsEnabled_ = false;
  bool mouseMiddle_ = false;
  bool mouseRight_ = false;
  double lastX_ = 0.0;
  double lastY_ = 0.0;

  bool toggleTrailsLatch_ = false;
  bool pauseLatch_ = false;
  bool focusLatch_ = false;
  bool focusClosestLatch_ = false;
  bool resetLatch_ = false;
  bool plusLatch_ = false;
  bool minusLatch_ = false;
  bool selectNextLatch_ = false;
  bool selectPrevLatch_ = false;
  float scrollDelta_ = 0.0f;
  RenderActions actions_;
};
