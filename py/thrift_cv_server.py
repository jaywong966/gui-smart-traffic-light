
from threading import Thread

from frames.frames import LoadStreams
from yolo_detector.detector import yolo_detector
from thrift_server import ThriftServer
from traffic_light import TrafficSignal



def pass_detected_images(detector, traffic, cv_server):
    while True:
        if detector.get_detect_status():
            current_detected_imgs = detector.get_detected_imgs()
            cv_server.update_imgs(current_detected_imgs)
            # get coord
            # pass coord into ROI class to output (input: corrd,ROI output: counting/list of box)
            # pass counting to cv_server.update_count

            current_detected_counts = detector.get_detected_counts()
            cv_server.update_location_info(current_detected_counts)

            traffic.set_location_info(current_detected_counts)
            current_light_signal = traffic.get_traffic_signal()
            current_countdown = traffic.get_countdown()
            cv_server.update_light_signal(current_light_signal ,current_countdown)


if __name__ == "__main__":

    cv_server = ThriftServer()
    cv_server.start()

    frames = LoadStreams()
    detector = yolo_detector(frames)
    detector.start()  # detector.detect()

    traffic = TrafficSignal()
    traffic.start()

    thread = Thread(target=pass_detected_images, args=([detector, traffic, cv_server]), daemon=True)
    print('start pass detected images')
    thread.start()


