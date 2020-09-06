import cv2
import numpy as np
import os
import json
import tkinter as tk
from tkinter import simpledialog

from SelectImageArea.SelectImageArea import ImageAreaSelection as ias
from frames.frames import LoadStreams , letterbox

if __name__ == '__main__':
    #Global variables
    ROOT = tk.Tk()
    ROOT.withdraw()
    module_path = os.path.dirname(os.path.abspath( __file__ ))
    mode = simpledialog.askinteger(title="Welcome to ROI selection helper(1/2)",
                                      prompt="which mode do you want?\n (0 for ROI mode, 1 for IOU mode, 2 for ROI for all mode, 3 for IOU for all mode):",
                                      initialvalue = 0)
    if mode ==0 or mode ==1:
        i = simpledialog.askinteger(title="Welcome to ROI selection helper(2/2)",
                                      prompt="which is your target?:",
                                      initialvalue = 0)
        
    if mode == 0 or mode == 1 or mode == 2 or mode == 3:
        frames = LoadStreams()
        imgs = frames.get_frames()
        
        if mode == 0:
            imgs = [letterbox(x, new_shape=frames.img_size, auto=frames.rect)[0] for x in imgs]
            cv2.imwrite('SelectImageArea/temp.jpg', imgs[i])
            inputImage = cv2.imread('SelectImageArea/temp.jpg')           
            roi_A = ias(inputImage)
            roi_A.letUserClickPoints()
            roi_A.ShowSelectedArea()
            with open('{}/detection_roi/config/preprocessing_config.txt'.format(module_path)) as json_file:
                preprocessing_config = json.load(json_file)
                preprocessing_config[str(i)] = f'{roi_A.mouseClickPoints}'
            with open('{}/detection_roi/config/preprocessing_config.txt'.format(module_path),'w') as json_file:
                json.dump(preprocessing_config ,json_file,indent = 4, sort_keys=True)
        elif mode == 1:
            cv2.imwrite('SelectImageArea/temp.jpg', imgs[i])
            inputImage = cv2.imread('SelectImageArea/temp.jpg')
            roi_A = ias(inputImage)
            roi_A.letUserClickPoints()
            roi_A.ShowSelectedArea()
            roi_B = ias(roi_A.imgWidthQuadrilaterals)
            roi_B.letUserClickPoints()
            roi_B.ShowSelectedArea()
            with open('{}/detection_roi/config/postprocessing_config.txt'.format(module_path)) as json_file:
                preprocessing_config = json.load(json_file)
                preprocessing_config[str(i)] = f'{roi_A.mouseClickPoints},{roi_B.mouseClickPoints}'
            with open('{}/detection_roi/config/postprocessing_config.txt'.format(module_path),'w') as json_file:
                json.dump(preprocessing_config ,json_file,indent = 4, sort_keys=True)
        elif mode == 2:
            imgs = [letterbox(x, new_shape=frames.img_size, auto=frames.rect)[0] for x in imgs]
            for i, img in enumerate(imgs):
                cv2.imwrite('SelectImageArea/temp.jpg', img)
                inputImage = cv2.imread('SelectImageArea/temp.jpg')           
                roi_A = ias(inputImage)
                roi_A.letUserClickPoints()
                roi_A.ShowSelectedArea()
                with open('{}/detection_roi/config/preprocessing_config.txt'.format(module_path)) as json_file:
                    preprocessing_config = json.load(json_file)
                    preprocessing_config[str(i)] = f'{roi_A.mouseClickPoints}'
                with open('{}/detection_roi/config/preprocessing_config.txt'.format(module_path),'w') as json_file:
                    json.dump(preprocessing_config ,json_file,indent = 4, sort_keys=True)
        elif mode == 3:
            for i, img in enumerate(imgs):
                cv2.imwrite('SelectImageArea/temp.jpg', img)
                inputImage = cv2.imread('SelectImageArea/temp.jpg')
                roi_A = ias(inputImage)
                roi_A.letUserClickPoints()
                roi_A.ShowSelectedArea()
                roi_B = ias(roi_A.imgWidthQuadrilaterals)
                roi_B.letUserClickPoints()
                roi_B.ShowSelectedArea()
                with open('{}/detection_roi/config/postprocessing_config.txt'.format(module_path)) as json_file:
                    preprocessing_config = json.load(json_file)
                    preprocessing_config[str(i)] = f'{roi_A.mouseClickPoints},{roi_B.mouseClickPoints}'
                with open('{}/detection_roi/config/postprocessing_config.txt'.format(module_path),'w') as json_file:
                    json.dump(preprocessing_config ,json_file,indent = 4, sort_keys=True)
                
    else:
        print("sorry, only mode0, mode1, mode2 and mode3 are available.")
    # More example
# =============================================================================
#     
#     cv2.namedWindow( f'Result' ,cv2.WINDOW_NORMAL)
#     imageWithQuadliterals = inputImage.copy()
#     #Draw quadliterals on the image
#     Point_length = len( myImageAreaSelector.mouseClickPoints )
#     #Make Point_length multiple of 4
#     Point_length =( Point_length >> 2 ) << 2
# 
#     for i in range( 0, Point_length, 4 ):
#         #cv2.polylines(imageWithQuadliterals, np.array([myImageAreaSelector.mouseClickPoints[i:i+4]]),True, (0,200,0),3)
#         points = np.array( [ myImageAreaSelector.mouseClickPoints[i:i+4] ], dtype=np.int32 )
#         cv2.fillPoly( imageWithQuadliterals, [points], (0, 200, 0, 0.5) )
#              
#     cv2.imshow( f'Result', imageWithQuadliterals )
#     cv2.waitKey(0)
#     cv2.destroyWindow( f'Result' )
# =============================================================================
    #cv2.destroyAllWindows()