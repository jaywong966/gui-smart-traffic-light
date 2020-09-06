# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 22:13:03 2020

@author: admin
"""
import random
import time
import os
import sys
from threading import Thread

module_path = os.path.dirname(os.path.abspath( __file__ ))
sys.path.append(module_path)
from utils.utils import *
from detection_roi.roi import roi_postprocessing , draw_roi, create_masks

class yolo_detector:
    
    def __init__(self, frames):
        
        self.module_path = os.path.dirname(os.path.abspath( __file__ ))
        self.model_config, self.inference_config, self.detect_config = load_config(self.module_path)
        self.device, self.model, self.modelc, self.coco_classes, self.custom_classes = load_model(self.module_path, self.model_config )
        
        self.frames = frames
        self.detected_imgs = [None] * self.frames.get_source_num()
        
        self.isdetect = False
        #Add 19/08/2020
        self.detected_counts = [None] * self.frames.get_source_num()
        self.iou_list = frames.iou_list
        self.postprocessing_flag = frames.postprocessing_flag
        self.draw_iou_flag = frames.draw_iou_flag
        #
    
  
    def detect(self):
        
        module_path = self.module_path
        frames = self.frames
        model = self.model
        modelc = self.modelc
        coco_classes = self.coco_classes
        custom_classes = self.custom_classes
        inference_config = self.inference_config
        detect_config = self.detect_config
        device = self.device
        #Add 19/08/2020
        iou_list = self.iou_list 
        #Add 26/08/2020
        black_image, ROImasksA, ROImasksB = create_masks(frames.get_frames(), iou_list)
        
        postprocessing_flag = self.postprocessing_flag
        draw_iou_flag = self.draw_iou_flag
        
        
        #read config
        out, view_img, save_img, save_txt, fourcc  = detect_config.values()
        half, conf_thres, iou_thres, augment, agnostic_nms, classify = inference_config.values()
    
        #init
        t0 = time.time()
        random.seed(30)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(coco_classes))]
        vid_writers = load_vid_writers(frames, out, fourcc, module_path)
    
        
        # Run detection
        for sources, imgs_to_model, imgs_to_show, vid_cap in frames:
                    
            #inference
            pred, imgs_to_model , imgs_to_show, Inference_time, NMS_time, Classifier_time = inference(device, imgs_to_model, imgs_to_show, model, modelc, inference_config)
            
            #Processing (Edited: add one more parameter(iou_list) for Process_detection)
            im0s_detection, detection_results, instances_of_classes, object_counts = Process_detections(module_path, out, sources, colors, pred, imgs_to_model, imgs_to_show, coco_classes, custom_classes, save_txt,postprocessing_flag, iou_list,black_image, ROImasksA, ROImasksB)          
            # im0s_detection = [draw_image(source, im0_detection, detection_result, Inference_time, NMS_time, Classifier_time) for source, im0_detection, detection_result in zip(sources, im0s_detection, detection_results)]
            self.detected_imgs = im0s_detection
            self.isdetect = True
            #Add 19/08/2020
            self.detected_counts = object_counts
            #
            
            # Save results (frames with detections)
            if save_img:
                [vid_writer.write(im0_detection) for vid_writer,im0_detection in zip(vid_writers,im0s_detection)]
                
            # Stream results (frames with detections)
            if view_img:
                if draw_iou_flag:
                    im0s_detection = [draw_roi(im0_detection,iou[0],(255,255,0)) for im0_detection, iou in zip(im0s_detection, iou_list)]
                [display_image(source, im0_detection, sources.index(source)) for source, im0_detection in zip(sources, im0s_detection)]
               
            # Print time (inference + NMS)
# =============================================================================
#             for instance_of_classes in instances_of_classes:  
#                 print('%sDone. (Inference %.3fs)(NMS_time %.3fs)(Classifier_time %.3fs)' % (instance_of_classes, Inference_time, NMS_time, Classifier_time))
# =============================================================================
        
        #release video writer
        if save_img:  
            [vid_writer.release() for vid_writer in vid_writers]

            
        #results.append(result_in_frame)
        print('Done. (%.3fs)' % (time.time() - t0)) 
        # print(self.get_detected_counts())
    
        if save_txt or save_img:
            print('Results saved to %s' % os.path.join(module_path,out))
    #Add 19/08/2020
    def get_detected_counts(self):
        return self.detected_counts
    #
    def get_detected_imgs(self):
        return self.detected_imgs

    def get_detect_status(self):
        return self.isdetect
    
    def start(self):
        thread = Thread(target=self.detect, daemon=True)
        print('start detection')
        thread.start()
        #thread.join()





# if __name__ == '__main__':
    
#     frames = LoadStreams()
#     detector = yolo_detector(frames)
#     detector.detect()