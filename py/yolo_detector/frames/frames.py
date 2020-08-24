# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 13:41:02 2020

@author: admin
"""


import os
import time
from threading import Thread
import cv2
import numpy as np
import json

class LoadStreams:  # multiple IP or RTSP cameras

    def __init__(self, sources='streams.txt', script_path= '', img_size=416):
        
        # config of frames instance
        module_path = os.path.dirname(os.path.abspath( __file__ ))
        with open('{}/config/frame_config.txt'.format(module_path)) as json_file:
            frame_config = json.load(json_file)
        source_list_relative_path, imgsz = frame_config.values()
        source_list_abs_path  = os.path.join(module_path, source_list_relative_path)
        sources = source_list_abs_path
        script_path = module_path
        img_size=imgsz
        
        
        
        self.img_size = img_size
        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = []
                for x in f.read().splitlines():
                    if len(x.strip()) and x.strip().endswith('.mp4'):
                        sources.append(os.path.join(script_path, x.strip()))
                    else:
                        sources.append(x.strip())
                #sources = [os.path.join(script_path, x.strip()) for x in f.read().splitlines() if len(x.strip())] ####
                #x.strip().endswith('.mp4')
        else:
            sources = [sources]
        
        n = len(sources)
        self.n = n
        self.imgs = [None] * n
        self.caps = [None] * n
        self.modes = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            if s.isdigit():
                self.modes[i] = 'webcam'
                cap = cv2.VideoCapture(int(s))
            elif s.startswith('http'):
                self.modes[i] = 'http'
                cap = cv2.VideoCapture(s)
            elif s.startswith('rtsp'):
                self.modes[i] = 'rtsp'
                cap = cv2.VideoCapture(s)
            elif s.endswith('.mp4'):
                self.modes[i] = 'mp4'
                cap = cv2.VideoCapture(s)
                
                
            print('%g/%g: %s... ' % (i + 1, n, s), end='')
            #cap = cv2.VideoCapture(0 if s == '0' else s)
            self.caps[i] = cap
            assert cap.isOpened(), 'Failed to open %s' % s
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            #thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            thread = Thread(target=update, args=([self.imgs, self.modes, i, self.caps[i]]), daemon=True)
            print(' success (%gx%g at %.2f FPS).' % (w, h, fps))
            thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, new_shape=self.img_size)[0].shape for x in self.imgs], 0)  # inference shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def display_image(self):
        1

    def get_soruce_num(self):
        return self.n

    def get_soruce_path(self):
        return self.sources

    def get_caps(self):
        return self.caps

    def get_frames(self):
        return self.imgs
    

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        
        self.imgs_to_show = self.imgs.copy()
        
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration
        # Letterbox
        imgs_to_model = [letterbox(x, new_shape=self.img_size, auto=self.rect)[0] for x in self.imgs_to_show]

        # Stack
        imgs_to_model = np.stack(imgs_to_model, 0)

        # Convert
        imgs_to_model = imgs_to_model[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416 
        imgs_to_model = np.ascontiguousarray(imgs_to_model)

        self.imgs_to_model = imgs_to_model

        return self.sources, self.imgs_to_model, self.imgs_to_show, self.caps

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years



def update(imgs, modes, index, cap):
    # Read next stream frame in a daemon thread
    n = 0
    frame_counter = 0

    while cap.isOpened():
        
        if modes[index] == "mp4":
            frame_counter += 1
            if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                frame_counter = 0 #Or whatever as long as it is the same as next line
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        n += 1
        # _, self.imgs[index] = cap.read()
        cap.grab()
        if n == 4:  # read every 4th frame
            ret_val, im = cap.retrieve()
            if ret_val:
                imgs[index] = im
            else:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                imgs[index] = np.zeros((h,w,3)).astype(np.uint8)
                    
            n = 0
        time.sleep(0.01)  # wait time


def letterbox(img, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


