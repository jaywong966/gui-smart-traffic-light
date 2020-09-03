# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os 
import json
import ast
import copy
import time

def get_roi_frame(current_frame, polygon):
    mask = np.zeros(current_frame.shape, dtype=np.uint8)
    polygon = np.array([polygon], dtype=np.int32)
    print(polygon)
    num_frame_channels = current_frame.shape[2]
    mask_ignore_color = (255,) * num_frame_channels
    cv2.fillPoly(mask, polygon, mask_ignore_color)
    masked_frame = cv2.bitwise_and(current_frame, mask)
    return masked_frame

def roi_preprocessing(imgs, roi_list):
    for img, polygon in zip(imgs, roi_list):
        if len(polygon) >= 3:
                index = roi_list.index(polygon)
                mask = np.zeros(img.shape, dtype=np.uint8)
                polygon = np.array(polygon, dtype=np.int32)
                num_frame_channels = img.shape[2]
                mask_ignore_color = (255,) * num_frame_channels
                cv2.fillPoly(mask, [polygon], mask_ignore_color)
                imgs[index] = cv2.bitwise_and(img, mask)
    return imgs

def roi_postprocessing(pred, imgs_to_model, iou_list): 
    mask = [None]* len(pred)
    for i, det in enumerate(pred):
        if len(iou_list[i][0]) >= 3 and len(iou_list[i][1]) >= 3  :
            if det is not None and len(det):
                polygonA = np.array([iou_list[i][0]], dtype=np.int32)
                polygonB = np.array([iou_list[i][1]], dtype=np.int32)
                # imshow all bbox (1/3)
                # all_bboxs = np.zeros(imgs_to_model[0].shape, dtype=np.uint8)
                # all_bboxs = all_bboxs[::-1, :, :].transpose(1, 2, 0) 
                # all_bboxs = np.ascontiguousarray(all_bboxs)
                for j , bbox in  enumerate(det[:, :4]):
                    #creat black image
                    mask[i] = np.zeros(imgs_to_model[0].shape, dtype=np.uint8)
                    mask[i] = mask[i][::-1, :, :].transpose(1, 2, 0) 
                    mask[i] = np.ascontiguousarray(mask[i])
                    ROImaskA = np.zeros(imgs_to_model[0].shape, dtype=np.uint8)
                    ROImaskA = ROImaskA[::-1, :, :].transpose(1, 2, 0) 
                    ROImaskA = np.ascontiguousarray(ROImaskA) 
                    ROImaskB = np.zeros(imgs_to_model[0].shape, dtype=np.uint8)
                    ROImaskB = ROImaskB[::-1, :, :].transpose(1, 2, 0) 
                    ROImaskB = np.ascontiguousarray(ROImaskB) 
                    a1, a2, a3, a4 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    #bbox 20%
                    a2 = a4 - round((a4- a2 )* 0.2)
                    c1, c2 = (a1,a2), (a3,a4)
                    cv2.rectangle(mask[i], c1, c2, (255,255,255), -1, cv2.LINE_AA)
                    # imshow all bbox (2/3)
                    # cv2.rectangle(all_bboxs, c1, c2, (255,255,255), -1, cv2.LINE_AA)
                    
                    # show before iou (each bbox)
                    # cv2.namedWindow( 'test1:' +str(j) , cv2.WINDOW_NORMAL) 
                    # cv2.imshow('test1:' +str(j), mask[i])
                    
                    
                    nonzero_count = len(mask[i][np.nonzero(mask[i])])
                    print('before')                    
                    print(nonzero_count)
                    cv2.fillPoly(ROImaskA, polygonA, (255,255,255))
                    cv2.fillPoly(ROImaskB, polygonB, (255,255,255))
                    # show roi
                    # cv2.namedWindow( 'ROI_A' , cv2.WINDOW_NORMAL) 
                    # cv2.imshow( 'ROI_A', ROImaskA)
                    # cv2.namedWindow( 'ROI_B' , cv2.WINDOW_NORMAL) 
                    # cv2.imshow( 'ROI_B', ROImaskB)
                    output_A = cv2.bitwise_and(mask[i],ROImaskA)
                    non_zero_on_A  = len(output_A[np.nonzero(output_A)])
                    output_B = cv2.bitwise_and(mask[i],ROImaskB)
                    non_zero_on_B = len(output_B[np.nonzero(output_B)])
                    print('after')                    
                    print(non_zero_on_A, non_zero_on_B)
                    if non_zero_on_A > non_zero_on_B:
                        print('{}{} is on the left hand side'.format(c1,c2) )
                        # pred.remove(str(pred[i][j]))
                    if non_zero_on_B > non_zero_on_A:
                        print( '{}{} is on the right hand side'.format(c1,c2) )

                    # imshow after iou
                    # cv2.namedWindow( 'test2A:' +str(j) , cv2.WINDOW_NORMAL) 
                    # cv2.imshow('test2A:' + str(j) , output_A)   
                    # cv2.namedWindow( 'test2B:' +str(j) , cv2.WINDOW_NORMAL) 
                    # cv2.imshow('test2B:' + str(j) , output_B)   

                # imshow all bbox(3/3)
                # cv2.namedWindow( 'all bbox on stream ' +str(i) , cv2.WINDOW_NORMAL) 
                # cv2.imshow('all bbox on stream ' +str(i)  , all_bboxs)  
                # show roi
                # cv2.namedWindow( 'ROI_A' , cv2.WINDOW_NORMAL) 
                # cv2.imshow( 'ROI_A', ROImaskA)
                # cv2.namedWindow( 'ROI_B' , cv2.WINDOW_NORMAL) 
                # cv2.imshow( 'ROI_B', ROImaskB)
                    
    return


def create_masks(imgs_to_show, iou_list):
    black_image = np.zeros(imgs_to_show[0].shape[:2], dtype=np.uint8)
    cv2.resize(black_image,None,fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    polygonsA = []
    polygonsB = []
    ROImasksA = []
    ROImasksB = []
    for x in iou_list:
        ROImasksA.append(copy.deepcopy(black_image))
        ROImasksB.append(copy.deepcopy(black_image))
        polygonsA.append((np.array(x[0])/4).astype("int32"))
        polygonsB.append((np.array(x[1])/4).astype("int32"))
    for polygonA, polygonB, ROImaskA, ROImaskB in zip(polygonsA, polygonsB, ROImasksA, ROImasksB):
        if len(polygonA) >= 3 :
            cv2.fillPoly(ROImaskA, [polygonA], (255,255,255))
        if len(polygonB) >= 3 :
            cv2.fillPoly(ROImaskB, [polygonB], (255,255,255))
    return black_image, ROImasksA, ROImasksB

def roi_postprocessing_by_xyxy(xyxy, imgs_to_show, stream_index , iou_list,  black_image, ROImasksA, ROImasksB):
    # Case1: ROI_A is empty, then do nothing.
    filter_flag = True
    t0 = time.time()
    # Case2: Both ROIs is not empty 
    if len(iou_list[stream_index][0]) >= 3 and len(iou_list[stream_index][1]) >= 3 :
        filter_flag = False
        
       #create black image
        mask = copy.deepcopy(black_image)
        t1 = time.time()
        
        #draw bbox 20% on the mask
        a1, a2, a3, a4 = int(xyxy[0]/4), int(xyxy[1]/4), int(xyxy[2]/4), int(xyxy[3]/4)
        a2 = a4 - round((a4- a2 )* 0.2)
        c1, c2 = (a1,a2), (a3,a4)
        cv2.rectangle(mask, c1, c2, (255,255,255), -1, cv2.LINE_AA)
        t2 = time.time()
        #get the amounts of non-zero pixels on ROI_A and ROI_B and then compare with them
        output_A = cv2.bitwise_and(mask,ROImasksA[stream_index])
        non_zero_on_A  = len(output_A[np.nonzero(output_A)])
        output_B = cv2.bitwise_and(mask,ROImasksB[stream_index])
        non_zero_on_B = len(output_B[np.nonzero(output_B)])
        t3 = time.time()
        t10 = t1-t0
        t21 = t2-t1
        t32 = t3-t2
        print ("     create black image: %.3f s" % t10)
        print("     create draw bbox on black image: %.3f s" % t21)
        print("     compare: %.3f s" % t32)
        #if non-zero pixel on ROI_A is more than on ROI_B, the bbox will be plotted and counted.
        if non_zero_on_A > non_zero_on_B:
            filter_flag = True
            
    # Case3: ROI_A is not empty but ROI_B is empty        
    if len(iou_list[stream_index][0]) >= 3 and len(iou_list[stream_index][1]) < 3 :
        filter_flag = False

        #create black image
        mask = copy.deepcopy(black_image)
       
        #draw bbox 20% on mask
        a1, a2, a3, a4 = int(xyxy[0]/4), int(xyxy[1]/4), int(xyxy[2]/4), int(xyxy[3]/4)
        a2 = a4 - round((a4- a2 )* 0.2)
        c1, c2 = (a1,a2), (a3,a4)       
        cv2.rectangle(mask, c1, c2, (255,255,255), -1, cv2.LINE_AA)    
        
        #get bbox size and compare with ROI
        nonzero_count = len(mask[np.nonzero(mask)])
        output_A = cv2.bitwise_and(mask,ROImasksA[stream_index])
        non_zero_on_A  = len(output_A[np.nonzero(output_A)])
        
        #check that bbox is on the ROI  
        if non_zero_on_A > nonzero_count*0.8:
            filter_flag = True
                
    return filter_flag

def draw_roi(frame, polygon,roi_bgr):
    if len(polygon) >= 3:
        frame_overlay = frame.copy()
        polygon = np.array([polygon], dtype=np.int32)
        cv2.fillPoly(frame_overlay, polygon, roi_bgr)
        alpha = 0.2
        frame_with_ROI = cv2.addWeighted(frame_overlay, alpha, frame, 1 - alpha, 0)
        return frame_with_ROI
    else:
        return frame

def load_roi_config(source_num):              
    roi_list = []
    iou_list = []
    module_path = os.path.dirname(os.path.abspath( __file__ ))
    with open('{}/config/preprocessing_config.txt'.format(module_path)) as json_file:
        preprocessing_config = json.load(json_file)
        
    with open('{}/config/flag_config.txt'.format(module_path)) as json_file:
        flag_config = json.load(json_file) 
        
    with open('{}/config/postprocessing_config.txt'.format(module_path)) as json_file:
        postprocessing_config = json.load(json_file)  
        
        
    for i in range(source_num):   
        try:
            roi_list.append(ast.literal_eval(preprocessing_config[str(i)]))
        except:
            pass 
    
    for j in range(source_num):   
        try:
            iou_list.append(ast.literal_eval(postprocessing_config[str(j)]))
        except:
            pass 
        
        
    preprocessing_flag = flag_config['preprocessing_flag']
    postprocessing_flag = flag_config['postprocessing_flag']
    draw_iou_flag = flag_config['draw_iou_flag']
    return roi_list, iou_list , preprocessing_flag, postprocessing_flag, draw_iou_flag
