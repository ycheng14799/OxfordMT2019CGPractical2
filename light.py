# light.py:
# Definition of lights
class PointLight(object):
    def __init__(self, position, intensity):
        self.position = position
        self.intensity = intensity
