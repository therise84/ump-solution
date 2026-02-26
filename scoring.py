import numpy as np
import cv2
import json
from math import sqrt

def compute_length(path):
    length = 0
    for i in range(len(path)-1):
        x1,y1 = path[i]
        x2,y2 = path[i+1]
        length += sqrt((x2-x1)**2 + (y2-y1)**2)
    return length

def compute_score(path, mask):
    length = compute_length(path)
    violations = 0
    for x,y in path:
        if mask[y,x] == 0:
            violations += 1
    score = 1000 - length - 50*violations
    return score, length, violations