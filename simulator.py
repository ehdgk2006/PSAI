import cv2
import numpy as np
import random
import math
import imageio


IMAGE_SIZE = (256, 256, 3)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)

class Circle:
    def __init__(self):
        self.radius = 20

        self.pos = [random.randint(self.radius, IMAGE_SIZE[0]-self.radius), random.randint(self.radius, IMAGE_SIZE[1]-self.radius)]
        self.velocity = [random.randint(-10, 10), random.randint(-10, 10)]
    

    def update(self):
        self.pos[0] += self.velocity[0]
        self.pos[1] += self.velocity[1]

        if self.pos[0] > IMAGE_SIZE[0] - self.radius:
            self.velocity[0] *= -1
            self.pos[0] = -self.pos[0] + 2*(IMAGE_SIZE[0] - self.radius)

        if self.pos[1] > IMAGE_SIZE[1] - self.radius:
            self.velocity[1] *= -1
            self.pos[1] = -self.pos[1] + 2*(IMAGE_SIZE[0] - self.radius)

        if self.pos[0] < self.radius:
            self.velocity[0] *= -1
            self.pos[0] = -self.pos[0] + self.radius
        
        if self.pos[1] < self.radius:
            self.velocity[1] *= -1
            self.pos[1] = -self.pos[1] + self.radius
    

    def get_img(self):
        img = np.zeros(IMAGE_SIZE, np.uint8)
        img.fill(0)
        img = cv2.circle(img, self.pos, self.radius, BLUE, cv2.FILLED, cv2.LINE_AA)

        return img


class Pendulum:
    def __init__(self):
        self.l = 100
        self.g = 1.
        
        self.angle = (random.random() * math.pi) - (math.pi / 2)
        self.angleA = 0
        self.angleV = 0

        self.pos = [self.l * math.sin(self.angle), self.l * math.cos(self.angle)]
        self.radius = 20
    

    def update(self):
        force = self.g * self.angle

        self.angleA = -force / self.l
        self.angleV += self.angleA
        self.angle += self.angleV

        self.pos[0] = self.l * math.sin(self.angle)
        self.pos[1] = self.l * math.cos(self.angle)
    

    def get_img(self):
        img = np.zeros(IMAGE_SIZE, np.uint8)
        img.fill(0)

        p1 = [int(IMAGE_SIZE[0] / 2), 0]
        p2 = [int(self.pos[0] + (IMAGE_SIZE[0] / 2)), int(self.pos[1])]

        img = cv2.line(img, p1, p2, BLACK, 2, cv2.LINE_AA)
        img = cv2.circle(img, p2, self.radius, WHITE, cv2.FILLED, cv2.LINE_AA)

        return img


def make_frames(obj, frames = None):
    res = []

    if frames == None:
        frames = random.randint(3, 10)
    # obj = Circle()
    
    for _ in range(frames):
        res.append(obj.get_img())
        obj.update()
    
    return res


if __name__ == '__main__':
    p = Pendulum()
    f = make_frames(p, 100)

    imageio.mimsave('./test.gif', f, duration=50)
