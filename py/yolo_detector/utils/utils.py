
from pathlib import Path
import random
import os
import time

import numpy as np
import json
import cv2

import torch
import torchvision
from utils import torch_utils
from detection_roi.roi import roi_postprocessing_by_xyxy

# Set printoptions
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
# matplotlib.rc('font', **{'size': 11})

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)





def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, multi_label=True, classes=None, agnostic=False):
    """
    Performs  Non-Maximum Suppression on inference results
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """

    # Settings
    merge = True  # merge for best mAP
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = conf_thres == 0.001  # require redundant detections

    t = time.time()
    nc = prediction[0].shape[1] - 5  # number of classes
    multi_label &= nc > 1  # multiple labels per box
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[x[:, 4] > conf_thres]  # confidence
        # x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]  # width-height

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero().t()
            x = torch.cat((box[i], x[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1)
            x = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)[conf > conf_thres]

        # Filter by class
        if classes:
            x = x[(j.view(-1, 1) == torch.tensor(classes, device=j.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5] * 0 if agnostic else x[:, 5]  # classes
        boxes, scores = x[:, :4].clone() + c.view(-1, 1) * max_wh, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def draw_image(p, imgs_to_show, r, Inference_time, NMS_time, Classifier_time):
    
    FPS = 1/(Inference_time + NMS_time + Classifier_time)
    position = (10,50)
    cv2.putText(
         imgs_to_show, #numpy array on which text is written
         "FPS: %.0f" % FPS , #text
         position, #position at which writing has to start
         cv2.FONT_HERSHEY_SIMPLEX, #font family
         1, #font size
         (0, 0, 255), #font color
         2) #font stroke
    
    position = (10,100)
    cv2.putText(
         imgs_to_show, #numpy array on which text is written
         r, #text
         position, #position at which writing has to start
         cv2.FONT_HERSHEY_SIMPLEX, #font family
         1, #font size
         (255, 0, 0), #font color
         2) #font stroke

    return imgs_to_show

def display_image(p, imgs_to_show, index):
    cv2.namedWindow(p, cv2.WINDOW_NORMAL) 
# =============================================================================
#     x = int((index%4)*400)
#     y = int((index /4)) * 300 + 150
#     cv2.moveWindow(p,x,y)
# =============================================================================
    cv2.imshow(p, imgs_to_show)   

def load_vid_writers(frames, out, fourcc, module_path):
    
    vid_path, vid_writers = [None] * frames.get_source_num(), [None] * frames.get_source_num()
    vid_paths = frames.get_source_path()
    vid_save_paths = [str(Path(out) / Path(vid_path).name) for vid_path in vid_paths]
    vid_caps = frames.get_caps()
    for i, vid_cap in enumerate(vid_caps):
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        abs_path = os.path.join(module_path, vid_save_paths[i])
        vid_writers[i] = cv2.VideoWriter(abs_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        
    return vid_writers

def inference(device, imgs_to_model, imgs_to_show, model, modelc, inference_config):
    
    half, conf_thres, iou_thres, augment, agnostic_nms, classify = inference_config.values()
    
    
    imgs_to_model = torch.from_numpy(imgs_to_model).to(device)
    imgs_to_model = imgs_to_model.half() if half else imgs_to_model.float()  # uint8 to fp16/32
    imgs_to_model /= 255.0  # 0 - 255 to 0.0 - 1.0
    if imgs_to_model.ndimension() == 3:
        imgs_to_model = imgs_to_model.unsqueeze(0)

    # Inference
    t1 = torch_utils.time_synchronized()
    pred = model(imgs_to_model, augment=augment)[0]
    t2 = torch_utils.time_synchronized()

    # to float
    if half:
        pred = pred.float()

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres,iou_thres,
                               multi_label=False, classes='', agnostic=agnostic_nms)
    t3 = torch_utils.time_synchronized()
    # Apply Classifier
    if classify:
        pred = apply_classifier(pred, modelc, imgs_to_model, imgs_to_show)
    t4 = torch_utils.time_synchronized()     
    
    
    Inference_time = t2-t1
    NMS_time = t3-t2
    Classifier_time = t4-t3
    return pred, imgs_to_model , imgs_to_show, Inference_time, NMS_time, Classifier_time




def Process_detections(module_path, out, paths, colors, pred, imgs_to_model, imgs_to_show, coco_classes, custom_classes, save_txt, postprocessing_flag,iou_list,black_image, ROImasksA, ROImasksB):
    
    im0s_detection = []
    detection_results = []
    instances_of_classes = []
    pedestrian_class = ['person']
    object_counts = []
    # start = time.time()
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        detection_result=''
        object_count = 0
        path, instance_of_classes, img_to_show = paths[i], '%g: ' % i, imgs_to_show[i]
        instance_of_classes += '%gx%g ' % imgs_to_model.shape[2:]  # print string
        gn = torch.tensor(img_to_show.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
        save_path_txt = str(Path(out[:out.rfind('/')] + os.sep + 'predictions') / Path(path).name)
        save_path_txt = os.path.join(module_path, save_path_txt)
        
        if det is not None and len(det):
            
            # Rescale boxes from imgs_to_model size to img_to_show size
            det[:, :4] = scale_coords(imgs_to_model.shape[2:], det[:, :4], img_to_show.shape).round()
            if i == (len(pred)-1):
                    custom_classes = pedestrian_class
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                instance_of_classes += '%g %ss, ' % (n, coco_classes[int(c)])  # add to string
                if any(elem in [coco_classes[int(c)]]  for elem in custom_classes):
                    detection_result += '%g %ss, ' % (n, coco_classes[int(c)])  # add to string
                
            # filter results
            for *xyxy, conf, cls in det:
                if any(elem in [coco_classes[int(cls)]]  for elem in custom_classes):
                    if postprocessing_flag:    
                        if roi_postprocessing_by_xyxy(xyxy, imgs_to_show ,i , iou_list, black_image, ROImasksA, ROImasksB):
                            label = 'Vehicle %.2f' % (conf)
                            if i == (len(pred)-1):  
                                label = 'Person %.2f' % (conf)
                            plot_one_box(xyxy, img_to_show, label=label, color=colors[int(cls)], line_thickness=1)
                            object_count += 1
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh               
                                with open(save_path_txt[:save_path_txt.rfind('.')] + '.txt', 'a') as file: #
                                    file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    else:
                        label = '%s %.2f' % (coco_classes[int(cls)], conf)
                        plot_one_box(xyxy, img_to_show, label=label, color=colors[int(cls)], line_thickness=1)
                        object_count += 1
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(save_path_txt[:save_path_txt.rfind('.')] + '.txt', 'a') as file: #
                                file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
        # end = time.time()      
        # total = end - start   
        # print("Finish detection process : %.3fs" % total)
        im0s_detection.append(img_to_show)
        detection_results.append(detection_result)
        instances_of_classes.append(instance_of_classes)
        object_counts.append(object_count)
        # print("This is detection_result:")
        # print(detection_results)        
        # print("This is instance_of_classes:")
        # print(instances_of_classes)
    return im0s_detection, detection_results, instances_of_classes, object_counts




def load_Second_stage_classifier():
    return


def load_model(module_path, model_config):
    
    device, weights, half, custom_class, target_class, classify  = model_config.values()
    
    weights = os.path.join(module_path, weights)
    custom_class = os.path.join(module_path, custom_class)
    target_class = os.path.join(module_path, target_class)

    
    modelc = None
    model = None
    # Initialize
    device = torch_utils.select_device('' if device=='cuda' else 'cpu' ) # cuda or cpu
    # Load model
    model = torch.load(weights, map_location=device)['model']
    
    # Eval mode
    model.to(device).eval()
    
    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()
         
    # Define classes
    #coco_classes = model.names
    coco_classes = open(target_class, "r").read().rstrip('\n').split('\n')
    custom_classes = coco_classes if not custom_class else open(custom_class, "r").read().rstrip('\n').split('\n')
    
    
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()
    
    
    return device, model, modelc, coco_classes, custom_classes


def load_config(module_path):
                
    with open(os.path.join(module_path,'config/model_config.txt')) as json_file:
        model_config = json.load(json_file)
        
    with open(os.path.join(module_path,'config/inference_config.txt')) as json_file:
        inference_config = json.load(json_file)
        
    with open(os.path.join(module_path,'config/detect_config.txt')) as json_file:
        detect_config = json.load(json_file)
        
    return model_config, inference_config, detect_config


def apply_classifier(x, model, img, im0):
    # applies a second stage classifier to yolo outputs
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                # cv2.imwrite('test%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x




# Plotting functions ---------------------------------------------------------------------------------------------------


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


