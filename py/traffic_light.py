import time
from numpy import *
from threading import Thread

class TrafficSignal:
    def __init__(self):
        self.vehicles_light = ["Red", "Red", "Red", "Red", "Red"]
        self.vehicles = 1
        self.other_vehicles = 0
        self.num_of_light = 0
        self.vehicles_collect = []
        self.other_vehicles_collect = []
        self.iscontrol = False
        self.isTimeout = True
        self.location_info = []
        self.countdown = 0
        #self.dector = yolo_detector()


    # def light_control(self):
    #     if self.vehicles == 0:
    #         self.num_of_light += 1
    #         while self.num_of_light == 5:
    #             self.num_of_light = 0
    #     elif self.vehicles < 2:
    #         if self.num_of_light < 4:
    #             self.isTimeout = False
    #             self.green_light(self.num_of_light)
    #             timer_green = Thread(target=self.timer_count(6))
    #             timer_green.start()
    #             self.yellow_light(self.num_of_light)
    #             timer_yellow = Thread(target=self.timer_count(3))
    #             timer_yellow.start()
    #             self.num_of_light += 1
    #             while self.num_of_light == 5:
    #                 self.num_of_light = 0
    #             self.isTimeout = True
    #         else:
    #             self.isTimeout = False
    #             self.green_light(self.num_of_light)
    #             timer_green = Thread(target=self.timer_count(7))
    #             timer_green.start()
    #             self.yellow_light(self.num_of_light)
    #             timer_yellow = Thread(target=self.timer_count(3))
    #             timer_yellow.start()
    #             self.num_of_light += 1
    #             while self.num_of_light == 5:
    #                 self.num_of_light = 0
    #             self.green_light(self.num_of_light)
    #             self.isTimeout = True
    #     elif self.vehicles > 3:
    #         self.green_light(self.num_of_light)
    #         print("vechile > 3")
        # if self.other_vehicles == 0:
        #     self.green_light(self.num_of_light)

    def light_control(self):
        if self.location_info[self.num_of_light] == 0:
            self.num_of_light += 1
            if self.num_of_light == 5:
                self.num_of_light = 0
        elif self.location_info[self.num_of_light] < 2:
            self.isTimeout = False
            self.green_light(self.num_of_light)
            timer_green = Thread(target=self.timer_count(5))
            timer_green.start()
            self.yellow_light(self.num_of_light)
            timer_yellow = Thread(target=self.timer_count(3))
            timer_yellow.start()
            self.num_of_light += 1
            if self.num_of_light == 5:
                self.num_of_light = 0
            if self.num_of_light != 3:
                self.green_light(self.num_of_light)
            self.isTimeout = True
        elif self.location_info[self.num_of_light] >= 3:
            self.green_light(self.num_of_light)


    def get_traffic_signal(self):
        return self.vehicles_light

    def get_countdown(self):
        return self.countdown

    def set_location_info(self,location_info):
        self.location_info = location_info
        self.isControl(self.location_info)
        if self.isTimeout:
            self.start()

    def isControl(self, location_info):
        self.vehicles = int(location_info[self.num_of_light])
        location_info = delete(location_info,self.num_of_light)
        self.other_vehicles = sum(location_info)
    # def isControl(self, location_info):
    #     if len(self.vehicles_collect) < 3:
    #         self.vehicles_collect.append(location_info[self.num_of_light])
    #         self.other_vehicles_collect.append(location_info[self.num_of_light])
    #         self.vehicles = int(mean(self.vehicles_collect))
    #         self.other_vehicles = sum(self.other_vehicles_collect)
    #         self.iscontrol = False
    #     else:
    #         self.iscontrol = True
    #         self.vehicles_collect.clear()
    #         self.other_vehicles_collect.clear()

    def green_light(self, number_of_light):
        for i in range(5):
            self.vehicles_light[i] = "Red"
        self.vehicles_light[number_of_light] = "Green"


    def yellow_light(self, number_of_light):
        for i in range(5):
            self.vehicles_light[i] = "Red"
        self.vehicles_light[number_of_light] = "Yellow"

    def start(self):
        thread = Thread(target=self.light_control ,daemon=True)
        thread.start()

    def timer_count(self,count):
        for i in range(count):
            self.countdown = count - i
            time.sleep(1)


if __name__ == '__main__':
    traffic = TrafficSignal()
    traffic.start()
    num = [1,0,1,0,1]
    while 1:
        traffic.set_location_info(num)
        print(traffic.vehicles_light)
        print(traffic.countdown)





