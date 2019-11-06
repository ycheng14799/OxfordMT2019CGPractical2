import numpy as np

class ray(object):
    def __init__(self, origin, position):
        self.origin = origin
        self.position = position

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


a = camera(pos = np.array([0.0,0.0,0.0]), to = np.array([0.0,1.0,0.0]),\
    upApprox = np.array([0.0,0.0,1.0]), focalLen = 1.0,\
    aspect = 4.0/3.0, widthActual = 3.0, widthPix = 400)
