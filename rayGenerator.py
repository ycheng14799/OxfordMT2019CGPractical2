import numpy as np

class camera(object):
    def __init__(self, pos, to, upApprox=np.array([0.0,0.0,1.0]),\
        focalLen, aspect):
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
        self.d = focalLen + 0.
