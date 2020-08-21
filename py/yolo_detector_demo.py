# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



from yolo_detector.detector import yolo_detector
from frames.frames import LoadStreams



if __name__ == '__main__':
    frames = LoadStreams()
    detector = yolo_detector(frames)
    detector.detect()