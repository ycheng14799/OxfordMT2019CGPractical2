# rayTracer.py:
# The actual ray tracing program

import numpy as np
import cv2
from rayGenerator import Camera, Ray
from geometry import Sphere, Triangle
from light import PointLight
import math

COLOR_CHANNELS = 3

def calcIntersectPoint(camPos, ray, t):
    rayDir = ray.direction / np.linalg.norm(ray.direction)
    return camPos + t * rayDir

def inShadow(scene, pointPos, lightPos, startOffset):
    # Calculate light vector
    l = lightPos - pointPos
    l = l / np.linalg.norm(l)
    # Test ray for intersection with scene objects
    lightRay = Ray(pointPos + (startOffset * l),  l)
    for obj in scene:
        intersection = obj.calcIntersection(lightRay)
        if intersection[0]:
            if intersection[1] > 0 and intersection[2] > 0:
                return True
    return False

def blinnPhongShadePoint(scene, I, color, ka, kd, ks, p, n, pointPos, camPos, lightPos):
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

    diffuse = np.multiply(kd * color, I) * max(0, np.dot(n, l))
    specular = np.multiply(ks * color, I) * max(0, np.dot(n, h))**p

    shade = np.multiply(ka * color, I) # Ambient

    if not inShadow(scene, pointPos, lightPos, 0.1): # Check in shadow
        shade = shade + diffuse + specular

    return shade

def reflect(normal, incident):
    return (incident - 2 * np.dot(incident, normal) * normal)

def rayColor(scene, start, lights, ray, maxReflect):
    intersected = False
    intersectT = np.inf
    intersectObj = scene[0]
    for obj in scene:
        intersectInfo = obj.calcIntersection(ray)
        if intersectInfo[0] == True:
            minIntersect = min(intersectInfo[1:])
            if minIntersect < intersectT and minIntersect > 0:
                intersectT = minIntersect
                intersected = True
                intersectObj = obj
    if intersected == True:
        intersectPoint = calcIntersectPoint(start, ray, intersectT)
        # Calculate intersection normal
        intersectionNormal = intersectObj.calcNormal(intersectPoint)
        # temporarily define a lighting position
        pointShade = 0.0
        for light in lights:
            pointShade += blinnPhongShadePoint(scene, light.intensity,\
                intersectObj.color, intersectObj.ka, intersectObj.kd,\
                intersectObj.ks, intersectObj.p, intersectionNormal,\
                intersectPoint, start, light.position)
        # calculate reflection
        reflectRayDir = reflect(intersectionNormal, ray.direction)
        reflectRay = Ray(intersectPoint, reflectRayDir)
        if maxReflect < 0:
            return pointShade
        else:
            return pointShade + rayColor(scene, intersectPoint, lights, reflectRay, maxReflect - 1)
    else:
        dir = ray.direction
        dir = dir / np.linalg.norm(dir)
        phi = (np.pi + np.arctan2(dir[1],dir[0])) / (2 * np.pi)
        theta = (-np.pi + np.arccos(dir[2])) / np.pi
        return map[int(theta * mapWidth), int(phi * mapHeight)]

# Scene definition
scene = []
'''
scene.append(Triangle(np.array([-2.0,-2.0,-1.0]),\
    np.array([-2.0,2.0,-1.0]),np.array([2.0,2.0,-1.0]),\
    np.array([1.0,1.0,1.0]), 0.25, 0.5, 0.5, 1))
scene.append(Triangle(np.array([-2.0,-2.0,-1.0]),\
    np.array([2.0,2.0,-1.0]),np.array([2.0,-2.0,-1.0]),\
    np.array([1.0,1.0,1.0]), 0.25, 0.5, 0.5, 1))
'''
scene.append(Sphere(np.array([0.0,2.0,0.0]),\
    1.0, np.array([1.0, 1.0, 1.0]), 0.25, 0.5, 0.5, 32))
scene.append(Sphere(np.array([0.0,-2.0,0.0]),\
    1.0, np.array([1.0, 1.0, 1.0]), 0.25, 0.5, 0.5, 32))

# Lights
lights = []
lights.append(PointLight(np.array([0.0, 0.0, 3.0]),\
    np.array([1.0, 1.0, 1.0])))

# Sphere map
map = cv2.imread("resources/autumnCubeMap.hdr", -1)
tonemapReinhard = cv2.createTonemapReinhard(1.5, 1.0,0.0,1.0)
map = tonemapReinhard.process(map)
mapHeight = map.shape[1]
mapWidth = map.shape[0]

# Camera
t = 0.0 # Camera control parameter
cam = Camera(np.array([10.0 * np.cos(t), 10.0 * np.sin(t), 0.0]),\
    np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),\
    20, 1.25, 25, 500)

# Screen
screenWidth = cam.widthPix
screenHeight = cam.heightPix
screen = np.zeros((screenHeight, screenWidth, COLOR_CHANNELS))

# Render loop
while(True):
    cam = Camera(np.array([10.0 * np.cos(t), 10.0 * np.sin(t), 0.0]),\
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),\
        20, 1.25, 25, 500)

    for i in range(0, screenHeight):
        for j in range(0, screenWidth):
            ray = cam.calcPixelPerpsectiveRay(j,i)
            screen[i,j] = rayColor(scene, cam.e, lights, ray, 2)

    cv2.imshow('frame',screen)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update camera control parameter
    t += 0.1

cv2.destroyAllWindows()
