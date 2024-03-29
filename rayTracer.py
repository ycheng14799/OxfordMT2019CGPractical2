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

def refract(normal, incident, n1, n2):
    n = n1 / n2
    cosI = -np.dot(normal, incident)
    sinT2 = n * n * (1.0 - (cosI * cosI))
    if sinT2 > 1.0:
        return False, np.zeros((3))
    cosT = np.sqrt(1.0 - sinT2)
    return True, n * incident + (n * cosI - cosT) * normal

def rSchlick(normal, incident, n1, n2):
    r0 = (n1 - n2) / (n1 + n2)
    r0 *= r0
    cosX = -np.dot(normal, incident)
    if (n1 > n2):
        n = n1 / n2
        sinT2 = n * n * (1.0 - cosX * cosX)
        if sinT2 > 1.0:
            return 1.0
        cosX = np.sqrt(1.0 - sinT2)
    x = 1.0 - cosX
    return r0 + (1.0 - r0) * x * x * x * x * x

def rayColor(scene, start, lights, ray, maxBounce, objRfractIdx):
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
        reflectRayDir = reflectRayDir / np.linalg.norm(reflectRayDir)
        reflectRay = Ray(intersectPoint, reflectRayDir)
        reflectionColor = rayColor(scene, intersectPoint, lights, reflectRay, maxBounce - 1, objRfractIdx)
        if maxBounce < 0:
            return 0.0 * pointShade #0.5 * pointShade
        else:
            if np.dot(intersectionNormal, ray.direction) < 0:
                notTIR, refractRayDir = refract(intersectionNormal, ray.direction, 1.0, objRfractIdx)
                R = rSchlick(intersectionNormal, ray.direction, 1.0, objRfractIdx)
                k = np.ones((3,))
            else:
                notTIR, refractRayDir = refract(-intersectionNormal, ray.direction, objRfractIdx, 1.0)
                R = rSchlick(-intersectionNormal, ray.direction, objRfractIdx, 1.0)
                k = np.exp([-0.1*intersectT, -0.1*intersectT, -0.1*intersectT])
            if notTIR == False:
                return np.multiply(k, reflectionColor)
            else:
                refractRay = Ray(intersectPoint, refractRayDir)
                refractionColor = rayColor(scene, intersectPoint, lights, refractRay, maxBounce - 1, objRfractIdx)
                return np.multiply(k, R * reflectionColor + (1.0 - R) * refractionColor)
    else:
        dir = ray.direction
        dir = dir / np.linalg.norm(dir)
        phi = (np.pi + np.arctan2(dir[1],dir[0])) / (2 * np.pi)
        theta = (-np.pi + np.arccos(dir[2])) / np.pi
        return map[int(theta * mapWidth), int(phi * mapHeight)]

# Scene definition

scene = []

scene.append(Sphere(np.array([0.0,0.0,0.0]),\
    2.0, np.array([1.0, 1.0, 1.0]), 0.0, 0.5, 0.5, 32))


#scene.append(Sphere(np.array([0.0,-2.0,0.0]),\
#    1.0, np.array([1.0, 1.0, 1.0]), 0.0, 0.5, 0.5, 32))

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
cam = Camera(np.array([4.0 * np.cos(t), 4.0 * np.sin(t), 0.0]),\
    np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),\
    20, 1.25, 25, 400)

# Screen
screenWidth = cam.widthPix
screenHeight = cam.heightPix
screen = np.zeros((screenHeight, screenWidth, COLOR_CHANNELS))

# Super-sampling count
numSample = 1

'''
# Write out video
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (screenWidth,screenHeight))

# Render loop
numFrames = 30
for frame in range(numFrames):
    cam = Camera(np.array([10.0 * np.cos(t), 10.0 * np.sin(t), 0.0]),\
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),\
        20, 1.25, 25, 500)
    # Render
    for i in range(0, screenHeight):
        for j in range(0, screenWidth):
            colorSum = np.zeros((3,))
            for sample in range(numSample):
                jSample = j + np.random.rand() - 0.5
                iSample = i + np.random.rand() - 0.5
                ray = cam.calcPixelPerpsectiveRay(jSample, iSample)
                colorSum += rayColor(scene, cam.e, lights, ray, 2)
            screen[i,j] = colorSum / numSample
    # Update camera control parameter
    t += 0.10
    out.write((screen * 256).astype('uint8'))

out.release()
'''
rfractIdxs = np.linspace(1.0, 1.1, 10)
for rfractIdx in rfractIdxs:

    cam = Camera(np.array([10.0 * np.cos(t), 10.0 * np.sin(t), 0.0]),\
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),\
        20, 1.25, 25, 400)
    # Render
    for i in range(0, screenHeight):
        for j in range(0, screenWidth):
            colorSum = np.zeros((3,))
            for sample in range(numSample):
                jSample = j + np.random.rand() - 0.5
                iSample = i + np.random.rand() - 0.5
                ray = cam.calcPixelPerpsectiveRay(jSample, iSample)
                colorSum += rayColor(scene, cam.e, lights, ray, 2, rfractIdx)
            screen[i,j] = colorSum / numSample
    cv2.imwrite("scene" + str(rfractIdx) + ".png", screen)

cv2.destroyAllWindows()
