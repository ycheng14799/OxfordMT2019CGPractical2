# rayTracer.py:
# The actual ray tracing program

import numpy as np
import matplotlib.pyplot as plt
import cv2
from rayGenerator import Camera, Ray
from geometry import Sphere, Triangle
from light import PointLight

COLOR_CHANNELS = 3

def calcIntersectPoint(camPos, ray, ts):
    t = min(ts)
    rayDir = ray.direction / np.linalg.norm(ray.direction)
    return camPos + t * rayDir

def blinnPhongShadePoint(I, ka, kd, ks, p, n, pointPos, camPos, lightPos):
    # I: Light intensity
    # ka, kd, ks: coefficients for ambient, diffuse, and specular Shading
    # p: Phong exponent
    # n: normal vector
    # pointPos: point position
    # camPos: camera position
    # lightPos: light position

    # calculate light vector
    l = lightPos - pointPos
    l = l / np.linalg.norm(l)
    # calculate view vector
    v = camPos - pointPos
    v = v / np.linalg.norm(v)
    # calculate halfway vector
    h = v + l
    h = h / np.linalg.norm(h)

    diffuse = np.multiply(kd, I) * max(0, np.dot(n, l))
    specular = np.multiply(ks, I) * max(0, np.dot(n, h))**p

    return np.multiply(ka, I) + diffuse + specular

# Scene definition
scene = []
scene.append(Sphere(np.array([2.0,0.0,1.0]),\
    1.0, np.array([1.0, 1.0, 1.0])))
scene.append(Triangle(np.array([0.0,-2.0,-1.0]),\
    np.array([0.0,2.0,-1.0]),\
    np.array([2.0,0.0,0.0]),\
    np.array([1.0,1.0,1.0])))
# Lights
lights = []
lights.append(PointLight(np.array([0.0, 3.0, 3.0]),\
    np.array([1.0, 1.0, 1.0])))

# Ray generation
cam = Camera(np.array([-2.0, 0.0, 0.0]),\
    np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),\
    1.0, 1.0, 4.0, 500)
viewRays = cam.calcPerspectiveRays()

# Ray intersection
screenWidth = cam.widthPix
screenHeight = cam.heightPix
screen = np.zeros((screenHeight, screenWidth, COLOR_CHANNELS))

for i in range(0, screenHeight):
    for j in range(0, screenWidth):
        for obj in scene:
            screenPos = (i * screenWidth) + j
            intersectInfo = obj.calcIntersection(viewRays[screenPos])
            if(intersectInfo[0] == True):
                #screen[i,j] = 1.0
                # Calculate shading
                # Calculate intersection point
                intersectPoint = calcIntersectPoint(cam.e, viewRays[screenPos], intersectInfo[1:])
                # Calculate intersection normal
                intersectionNormal = obj.calcNormal(intersectPoint)
                # temporarily define a lighting position
                pointShade = 0.0
                for light in lights:
                    pointShade += blinnPhongShadePoint(light.intensity,\
                        np.array([0.25, 0.25, 0.25]),\
                        np.array([0.75, 0.75, 0.75]),\
                        np.array([0.5, 0.5, 0.5]),\
                        1, intersectionNormal, intersectPoint,\
                        cam.e, light.position)
                screen[i,j] = pointShade


# Display screen
cv2.imshow('rendered', screen)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Shading

print(screen)
