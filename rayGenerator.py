import numpy as np

class ray(object):
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

class camera(object):
    def __init__(self, pos, to, upApprox,\
        focalLen, aspect, widthActual, widthPix):
        self.e = pos # Eye point
        self.e += 0. # Adding zero to avoid negative zeros issue with floats
        self.w = (pos - to) / np.linalg.norm(pos - to) # w vector, negative of forward
        self.w += 0.
        u = np.cross(upApprox, self.w)
        self.u = u / np.linalg.norm(u) # u vector, right
        self.u += 0.
        v = np.cross(self.w, u)
        self.v = v / np.linalg.norm(v) # v vector, up
        self.v += 0.
        self.d = focalLen + 0. # Focal length
        self.aspect = aspect + 0. # Aspect ratio
        self.width = widthActual # Actual width of camera plane
        self.height = widthActual / aspect # Actual height of camera plane
        self.widthPix = widthPix # Number of pixels across width
        self.heightPix = int(widthPix / aspect) # Number of pixels across height
        self.l = - (0.5 * self.width) # Left boundary of screen
        self.r = 0.5 * self.width # Right boundary of screne
        self.b = - (0.5 * self.height) # Bottom boundary of scene
        self.t = 0.5 * self.height # Top boundary of scene

    # Calculate orthographic ray through given (i,j) pixel
    def calcPixelOrthoRay(self, i, j):
        u = self.l + (self.r - self.l) * (i + 0.5) / self.widthPix
        v = self.b + (self.t - self.b) * (j + 0.5) / self.heightPix
        rayDirection = - self.w
        rayOrigin = self.e + (u * self.u) + (v * self.v)
        return ray(rayOrigin, rayDirection)

    # Rays for orthographic projection
    def calcOrthoRays(self):
        rays = []
        for i in range(0, self.widthPix):
            for j in range(0, self.heightPix):
                rays.append(self.calcPixelOrthoRay(i,j))
        return rays

    # Calculate perspective ray given (i,j) pixel
    def calcPixelPerpsectiveRay(self, i, j):
        u = self.l + (self.r - self.l) * (i + 0.5) / self.widthPix
        v = self.b + (self.t - self.b) * (j + 0.5) / self.heightPix
        rayOrigin = self.e
        rayDirection = -(self.d * self.w) + (u * self.u) + (v * self.v)
        rayDirection = rayDirection #/ np.linalg.norm(rayDirection)
        return ray(rayOrigin, rayDirection)

    # Rays for perspective projection
    def calcPerspectiveRays(self):
        rays = []
        for i in range(0, self.widthPix):
            for j in range(0, self.heightPix):
                rays.append(self.calcPixelPerpsectiveRay(i,j))
        return rays

a = camera(pos = np.array([0.0,0.0,0.0]), to = np.array([0.0,1.0,0.0]),\
    upApprox = np.array([0.0,0.0,1.0]), focalLen = 1.0,\
    aspect = 1.0, widthActual = 4.0, widthPix = 2)
