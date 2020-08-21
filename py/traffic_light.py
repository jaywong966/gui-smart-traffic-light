import time
from threading import Thread
from numpy import *

class TrafficSignal:
    def __init__(self):
        self.vehicles_light = ["Red", "Red", "Red", "Red", "Red"]
        self.vehicles = 0
        self.other_vehicles = 0
        self.pedestrians = 0
        self.vehicles_collect = []
        self.other_vehicles_collect = []
        self.iscontrol = False
        self.custom_class = ["cars", "trucks", "persons", "bicycles", "motorcycles", "busses", "traffic lights"]

    def light_control(self, num_of_light,location_info):
        self.gereen_light(num_of_light)
        self.verification(num_of_light, location_info)
        self.print_status()
        if self.vehicles < 3:
            if self.other_vehicles == 0:
                self.gereen_light(num_of_light)
                return self.vehicles_light
            time.sleep(5)
            return self.vehicles_light
        elif self.vehicles >= 3:
            self.gereen_light(num_of_light)
            return self.vehicles_light


    def verification(self, num_of_light, location_info):
        while len(self.vehicles_collect) < 3:
            object = location_info[num_of_light]
            self.vehicles_collect.append(object)
            for i in range(5):
                if i!=num_of_light:
                    self.other_vehicles_collect.append(location_info[i])
        self.vehicles = mean(self.vehicles_collect)
        self.other_vehicles = sum(self.other_vehicles_collect)
        self.iscontrol = True

    def print_status(self):
        for i in range(5):
            print("vehicle", str(i), self.vehicles_light[i])


    def gereen_light(self, number_of_light):
        for i in range(5):
            self.vehicles_light[i] = "Red"
        self.vehicles_light[number_of_light] = "Green"


    def start(self):
        thread = Thread(target=self.light_control, daemon=True)
        thread.start()

