# rayTracer.py:
# The actual ray tracing program

import numpy as np
from rayGenerator import Ray, Camera
from geometry import Sphere

def main():
    s = Sphere(np.array([0.0, 0.0, 0.0]), 1.0)
    ray = Ray(np.array([-2.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    intersectInfo = s.calcIntersection(ray)

if __name__ == "__main__":
    main()
