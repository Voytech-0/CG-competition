#include "scene.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>

#include "hit.h"
#include "image.h"
#include "material.h"
#include "ray.h"

using namespace std;

pair <ObjectPtr, Hit> Scene::castRay(Ray const &ray) const {
  // Find hit object and distance
  Hit min_hit(numeric_limits<double>::infinity(), Vector());
  ObjectPtr obj = nullptr;
  for (unsigned idx = 0; idx != objects.size(); ++idx) {
    Hit hit(objects[idx]->intersect(ray));
    if (hit.t < min_hit.t) {
      min_hit = hit;
      obj = objects[idx];
    }
  }

  return pair<ObjectPtr, Hit>(obj, min_hit);
}

Color Scene::trace(Ray const &ray, unsigned depth, const bool isInside = false) {
  pair <ObjectPtr, Hit> mainhit = castRay(ray);
  ObjectPtr obj = mainhit.first;
  Hit min_hit = mainhit.second;

  // No hit? Return background color.
  if (!obj) {
    return Color(0.0, 0.0, 0.0);
  }

  Material const &material = obj->material;
  Point hit = ray.at(min_hit.t);
  Vector V = -ray.D.normalized();

  // Pre-condition: For closed objects, N points outwards.
  Vector N = min_hit.N.normalized();

  // The shading normal always points in the direction of the view,
  // as required by the Phong illumination model.
  Vector shadingN;
  if (N.dot(V) >= 0.0)
    shadingN = N;
  else
    shadingN = -N;

  Color matColor = material.color;

  // Add ambient once, regardless of the number of lights.
  Color color = material.ka * matColor;

  // Add diffuse and specular components.
  for (auto const &light: lights) {
    Vector L = (light->position - hit).normalized();

    if (renderShadows) {
      Ray shadowRay(hit + epsilon * L.normalized(), L);
      pair <ObjectPtr, Hit> shadowHit = castRay(shadowRay);
      if (shadowHit.first != nullptr &&
          shadowHit.second.t < (light->position - hit).length()) {
        continue;
      }
    }

    // Add diffuse.
    double dotNormal = shadingN.dot(L);
    double diffuse = std::max(dotNormal, 0.0);
    color += diffuse * material.kd * light->color * matColor;

    // Add specular.
    if (dotNormal > 0) {
      Vector reflectDir = reflect(-L, shadingN);
      double specAngle = std::max(reflectDir.dot(V), 0.0);
      double specular = std::pow(specAngle, material.n);

      color += specular * material.ks * light->color;
    }
  }

  if (depth > 0 && material.isTransparent) {
    // The object is transparent, and thus refracts and reflects light.
    // Use Schlick's approximation to determine the ratio between the two.
    double nt = material.nt;
    double k_r0 = pow((1 - nt) / (1 + nt), 2);
    double k_r = k_r0 + (1 - k_r0) * pow(1 - shadingN.dot(V), 5);
    double k_t = 1 - k_r;
    if (k_r > 0.0) {
      Color reflectedColor =
          trace(Ray(hit + epsilon * shadingN, reflect(-V, shadingN).normalized()), depth - 1);
      color += k_r * reflectedColor;
    }

    if (k_t > 0.0) {
      Vector T;
      if (!isInside) {
        T = refract(-V, shadingN, 1, nt).normalized();
      } else {
        T = refract(-V, shadingN, nt, 1).normalized();
      }
      // check if the ray is inside the object
      if (T.dot(shadingN) < 0) {
        // the ray moves through the medium
        Ray insideRay = Ray(hit - epsilon * shadingN, T);
        color += k_t * trace(insideRay, depth - 1, !isInside);
      } else {
        // the ray does not change the medium - total internal reflection
        color += k_t * trace(Ray(hit + epsilon * shadingN, reflect(-V, shadingN).normalized()), depth - 1, isInside);
      }

//      pair<ObjectPtr, Hit> transHit = castRay(insideRay);
//      ObjectPtr obj2 = transHit.first;
//      Hit min_hit_t = transHit.second;
//      Point p = insideRay.at(min_hit_t.t);
//      Vector N2 = min_hit_t.N.normalized();
//      // make sure that N2 points inwards
//      if (N2.dot(T) >= 0) {
//        N2 = -N2;
//      }
//      Vector outgoing = refract(T, N2, nt, 1.0).normalized();
//      Color transmittedColor = trace(Ray(p - epsilon * N2, outgoing), depth - 1);
//      color += k_t * transmittedColor;
    }

  } else if (depth > 0 && material.ks > 0.0) {
    // The object is not transparent, but opaque.
    //if white, then there is no point in tracing further
    if (!(color.x >= 255 && color.y >= 255 && color.z >= 255)) {
      Color reflectedColor = trace(Ray(hit + epsilon * shadingN, reflect(-V, shadingN).normalized()), depth - 1);
      color += material.ks * reflectedColor; // + (1 - material.ks) * color;
    }
  }

  return color;
}

void Scene::render(Image &img) {
  unsigned w = img.width();
  unsigned h = img.height();

  Image superSampled = Image(w * supersamplingFactor, h * supersamplingFactor);


  for (unsigned y = 0; y < h * supersamplingFactor; ++y)
    for (unsigned x = 0; x < w * supersamplingFactor; ++x) {
      Point pixel((x + 0.5) / supersamplingFactor / (w / 400),
                  (h * supersamplingFactor - 1 - y + 0.5) / supersamplingFactor / (h / 400), 0);
      Ray ray(eye, (pixel - eye).normalized());
      Color col = trace(ray, recursionDepth);
      col.clamp();
      superSampled(x, y) = col;
    }
  for (unsigned y = 0; y < h; ++y)
    for (unsigned x = 0; x < w; ++x) {
      Color col(0.0, 0.0, 0.0);
      for (unsigned i = 0; i < supersamplingFactor; ++i)
        for (unsigned j = 0; j < supersamplingFactor; ++j)
          col += superSampled(x * supersamplingFactor + i, y * supersamplingFactor + j);
      col /= supersamplingFactor * supersamplingFactor;
      img(x, y) = col;
    }
}

// --- Misc functions ----------------------------------------------------------

// Defaults
Scene::Scene()
    : objects(), lights(), eye(), renderShadows(false), recursionDepth(0),
      supersamplingFactor(1) {}

void Scene::addObject(ObjectPtr obj) { objects.push_back(obj); }

void Scene::addLight(Light const &light) {
  lights.push_back(LightPtr(new Light(light)));
}

void Scene::setEye(Triple const &position) { eye = position; }

unsigned Scene::getNumObject() { return objects.size(); }

unsigned Scene::getNumLights() { return lights.size(); }

void Scene::setRenderShadows(bool shadows) { renderShadows = shadows; }

void Scene::setRecursionDepth(unsigned depth) { recursionDepth = depth; }

void Scene::setSuperSample(unsigned factor) { supersamplingFactor = factor; }
