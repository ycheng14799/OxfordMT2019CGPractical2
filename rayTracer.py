# rayTracer.py:
# The actual ray tracing program

import numpy as np
import matplotlib.pyplot as plt
from rayGenerator import Camera, Ray
from geometry import Sphere, Triangle

# Scene definition
scene = []
scene.append(Sphere(np.array([0.0,0.0,0.0]),\
    1.0, np.array([0.0, 0.0, 1.0])))

# Ray generation
cam = Camera(np.array([-2.0, 0.0, 0.0]),\
    np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),\
    1.0, 1.0, 4.0, 20)
viewRays = cam.calcOrthoRays()

# Ray intersection
screenWidth = cam.widthPix
screenHeight = cam.heightPix
screen = np.zeros((screenHeight, screenWidth))

for i in range(0, screenHeight):
    for j in range(0, screenWidth):
        for obj in scene:
            intersectInfo = obj.calcIntersection(viewRays[(i * screenWidth) + j])
            if(intersectInfo[0] == True):
                screen[i,j] = 1.0
plt.figure()
plt.imshow(screen)
# Shading

print(screen)
