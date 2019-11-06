# geometry.py:
# Definitions for geometry (sphere, triangle)

import numpy as np
import rayGenerator

class Sphere(object):
    def __init__(self, center, radius):
        self.c = center # Sphere center
        self.R = radius # Sphere radius

    # Method for calculating intersections
    def calcIntersection(self, ray):
        # Calculate discriminant first
        discrim = np.dot(-ray.direction, ray.origin - self.c)**2 \
            - np.dot(ray.direction, ray.direction) * \
            (np.dot(ray.origin - self.c, ray.origin - self.c) - self.R**2)
        if discrim < 0:
            return (False, np.full((3,),np.inf))
        else:
            t1 = (- np.dot(ray.direction, ray.origin - self.c) + discrim)\
                / np.dot(ray.direction, ray.direction)
            t2 = (- np.dot(ray.direction, ray.origin - self.c) - discrim)\
                / np.dot(ray.direction, ray.direction)
            return (True, t1, t2)
