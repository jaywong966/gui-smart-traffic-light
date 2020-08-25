# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 13:44:57 2020

@author: admin
"""


import cv2
import numpy as np

from frames import LoadStreams

if __name__ == '__main__':
    
    
    frames = LoadStreams()
    
    for sources, imgs_to_model, imgs_to_show, vid_cap in frames:
        for source, img_to_model, img_to_show in zip(sources, imgs_to_model, imgs_to_show):
            
            
            cv2.namedWindow(source, cv2.WINDOW_NORMAL) 
            cv2.imshow(source, img_to_show)  
            
            img_to_model = img_to_model[::-1, :, :].transpose(1, 2, 0) 
            img_to_model = np.ascontiguousarray(img_to_model)
            cv2.namedWindow(source+"(to_model)", cv2.WINDOW_NORMAL) 
            cv2.imshow(source+"(to_model)", img_to_model)  
    
    