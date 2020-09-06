# -*- coding: utf-8 -*-

import cv2
import numpy as np
from frames.frames import LoadStreams , letterbox
from detection_roi.roi import roi_postprocessing , draw_roi, create_masks

if __name__ == '__main__':
    frames = LoadStreams()
    imgs = frames.get_frames()
    iou_list = frames.iou_list
    x1, x2, x3 ,x4,x5= create_masks(imgs, iou_list)
    cv2.namedWindow( 'test1:' , cv2.WINDOW_NORMAL) 
    cv2.imshow('test1:', x5[0])
    cv2.waitKey(0) 
    cv2.destroyWindow('test1:')