import numpy as np

class Mode():
    color_detect = 0
    task_detect = 1

class Object_Data():
    position_matrix = np.zeros((4, 2), dtype=int)
    center = (0,0)
    color = 0

class WiFi_Scan:
    task_number = [0,0,0,0,0,0]