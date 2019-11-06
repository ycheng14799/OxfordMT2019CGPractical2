# rayTracer.py:
# The actual ray tracing program

import numpy as np
import rayGenerator
import geometry

def main():
    s = Sphere(np.array([0.0, 0.0, 0.0]), 1.0)
    ray = Ray(np.array([-2.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    intersectInfo = ray.calcIntersection(ray)
    print(intersectInfo[0])

if __name__ == "__main__":
    main()
