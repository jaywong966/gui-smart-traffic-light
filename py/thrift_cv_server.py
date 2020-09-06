

from threading import Thread

from frames.frames import LoadStreams
from yolo_detector.detector import yolo_detector
from thrift_server import ThriftServer



def pass_detected_images(detector, cv_server):
    while True:
        if detector.get_detect_status():
            current_detected_imgs = detector.get_detected_imgs()
            current_detected_counts = detector.get_detected_counts()
            cv_server.update_counts(current_detected_counts)
            cv_server.update_imgs(current_detected_imgs)
        
    
if __name__ == "__main__":

    cv_server = ThriftServer()
    cv_server.start()
    
    frames = LoadStreams()
    detector = yolo_detector(frames)
    detector.start()     #detector.detect()
    
    
    thread = Thread(target=pass_detected_images, args=([detector, cv_server]), daemon=True)
    print('start pass detected images')
    thread.start()
    
    
