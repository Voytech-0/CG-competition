#ifndef SCENE_H_
#define SCENE_H_

#include <utility>
#include <vector>

#include "light.h"
#include "object.h"
#include "triple.h"

// Forward declarations
class Ray;
class Image;

class Scene {
  std::vector<ObjectPtr> objects;
  std::vector<LightPtr> lights;
  Point eye;
  bool renderShadows;
  unsigned recursionDepth;
  unsigned supersamplingFactor;

  // Offset multiplier. Before casting a new ray from a hit point,
  // move the hit point in the direction of the normal with this offset
  // to prevent finding an intersection with the same object due to
  // floating point inaccuracies. This prevents shadow acne, among other problems.
  double const epsilon = 1E-3;

 public:
  Scene();

  // determine closest hit (if any)
  std::pair<ObjectPtr, Hit> castRay(Ray const &ray) const;

  // trace a ray into the scene and return the color
  Color trace(Ray const &ray, unsigned depth, const bool isInside);

  // render the scene to the given image
  void render(Image &img);

  void addObject(ObjectPtr obj);
  void addLight(Light const &light);
  void setEye(Triple const &position);
  void setRenderShadows(bool renderShadows);
  void setRecursionDepth(unsigned depth);
  void setSuperSample(unsigned factor);

  unsigned getNumObject();
  unsigned getNumLights();
};

#endif
