# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os 
import json
import ast
       
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
        if polygon is not None:
                index = roi_list.index(polygon)
                mask = np.zeros(img.shape, dtype=np.uint8)
                polygon = np.array([polygon], dtype=np.int32)
                num_frame_channels = img.shape[2]
                mask_ignore_color = (255,) * num_frame_channels
                cv2.fillPoly(mask, polygon, mask_ignore_color)
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

def roi_postprocessing_by_xyxy(xyxy, imgs_to_show,stream_index , iou_list):
    filter_flag = True
    if len(iou_list[stream_index][0]) >= 3 and len(iou_list[stream_index][1]) >= 3 :
        filter_flag = False
        polygonA = np.array([iou_list[stream_index][0]], dtype=np.int32)
        polygonB = np.array([iou_list[stream_index][1]], dtype=np.int32)
        # imshow all bbox (1/3)
        # all_bboxs = np.zeros(imgs_to_model[0].shape, dtype=np.uint8)
        # all_bboxs = all_bboxs[::-1, :, :].transpose(1, 2, 0) 
        # all_bboxs = np.ascontiguousarray(all_bboxs)
        #creat black image
        mask = np.zeros(imgs_to_show[0].shape, dtype=np.uint8)
        ROImaskA = np.zeros(imgs_to_show[0].shape, dtype=np.uint8)
        ROImaskB = np.zeros(imgs_to_show[0].shape, dtype=np.uint8)

        a1, a2, a3, a4 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        #bbox 20%
        a2 = a4 - round((a4- a2 )* 0.2)
        c1, c2 = (a1,a2), (a3,a4)
        cv2.rectangle(mask, c1, c2, (255,255,255), -1, cv2.LINE_AA)
        # imshow all bbox (2/3)
        # cv2.rectangle(all_bboxs, c1, c2, (255,255,255), -1, cv2.LINE_AA)
        
        # show before iou (each bbox)  (need update)
        # cv2.namedWindow( 'test1:' , cv2.WINDOW_NORMAL) 
        # cv2.imshow('test1:', ROImaskA)
        
        # # check bbox size
        # nonzero_count = len(mask[np.nonzero(mask)])
        # print('before')                    
        # print(nonzero_count)
        
        cv2.fillPoly(ROImaskA, polygonA, (255,255,255))
        cv2.fillPoly(ROImaskB, polygonB, (255,255,255))
        # show roi
        # cv2.namedWindow( 'ROI_A' , cv2.WINDOW_NORMAL) 
        # cv2.imshow( 'ROI_A', ROImaskA)
        # cv2.namedWindow( 'ROI_B' , cv2.WINDOW_NORMAL) 
        # cv2.imshow( 'ROI_B', ROImaskB)
        output_A = cv2.bitwise_and(mask,ROImaskA)
        non_zero_on_A  = len(output_A[np.nonzero(output_A)])
        output_B = cv2.bitwise_and(mask,ROImaskB)
        non_zero_on_B = len(output_B[np.nonzero(output_B)])
        # print('after')                    
        # print(non_zero_on_A, non_zero_on_B)
        if non_zero_on_A > non_zero_on_B:
            filter_flag = True
        # imshow after iou (need update)
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
                
    return filter_flag

def draw_roi(frame, polygon,roi_bgr):
        frame_overlay = frame.copy()
        polygon = np.array([polygon], dtype=np.int32)
        cv2.fillPoly(frame_overlay, polygon, roi_bgr)
        alpha = 0.2
        frame_with_ROI = cv2.addWeighted(frame_overlay, alpha, frame, 1 - alpha, 0)
        return frame_with_ROI

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
    return roi_list, iou_list , preprocessing_flag, postprocessing_flag
