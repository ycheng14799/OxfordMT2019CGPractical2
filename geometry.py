# geometry.py:
# Definitions for geometry (sphere, triangle)

import numpy as np
from rayGenerator import Ray

class Sphere(object):
    def __init__(self, center, radius, color):
        self.c = center + 0. # Sphere center
        self.R = radius + 0. # Sphere radius
        self.color = color + 0. # Sphere color

    # Method for calculating normals
    def calcNormal(self, p):
        n = (p - self.c) / self.R
        return n

    # Method for calculating intersections
    def calcIntersection(self, ray):
        # Calculate discriminant first
        discrim = np.dot(-ray.direction, ray.origin - self.c)**2 \
            - np.dot(ray.direction, ray.direction) * \
            (np.dot(ray.origin - self.c, ray.origin - self.c) - self.R**2)
        if discrim < 0:
            return (False, np.inf, np.inf)
        else:
            t1 = (- np.dot(ray.direction, ray.origin - self.c) + discrim)\
                / np.dot(ray.direction, ray.direction)
            t2 = (- np.dot(ray.direction, ray.origin - self.c) - discrim)\
                / np.dot(ray.direction, ray.direction)
            return (True, t1, t2)

class Triangle(object):
    def __init__(self, a, b, c, color): # Triangle with vertices a, b, c
        self.a = a + 0.
        self.b = b + 0.
        self.c = c + 0.
        self.color = color + 0. # Triangle color

    # Method for calculating intersections
    def calcIntersection(self, ray):
        # Write out systems of equations
        sysA = np.array([self.a - self.b, self.a - self.c, ray.direction]).T
        sysb = (self.a - ray.origin).T

        # Calculate M
        M = np.linalg.det(sysA)
        if M == 0:
            return (False, np.inf)

        # Calculate t
        tMat = np.copy(sysA)
        tMat[:,2] = sysb
        t = np.linalg.det(tMat) / M

        # Calculate gamma:
        gammaMat = np.copy(sysA)
        gammaMat[:,1] = sysb
        gamma = np.linalg.det(gammaMat) / M
        if (gamma < 0) or (gamma > 1):
            return (False, np.inf)

        # Calculate beta:
        betaMat = np.copy(sysA)
        betaMat[:,0] = sysb
        beta = np.linalg.det(betaMat) / M
        if (beta < 0) or (beta > (1 - gamma)):
            return (False, np.inf)

        return (True, t)
