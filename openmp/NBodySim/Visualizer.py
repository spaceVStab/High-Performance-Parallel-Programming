#!/usr/bin/env python

import numpy as np 
import pptk 
import time

with open('Trajectory.txt','r') as f_:
    po = f_.readlines()

poi = []
for i, l in enumerate(po):
    if i < 8:
        continue
    poi.append(np.array(l.strip().split("\t")))
poi = np.array(poi)

p = pptk.viewer(poi)
p.set(point_size=0.005)

def viz(points):
    points = np.array(points)
    p.load(points)
    p.set(point_size=0.005)


with open('FinalPositions_8.txt','r') as fp:
    points = []
    for i,lines in enumerate(fp):
        if lines is "\n":
            viz(points)
            time.sleep(.5)
            points = []
        else:
            point = lines.strip().split(" ")
            points.append(np.array(point))