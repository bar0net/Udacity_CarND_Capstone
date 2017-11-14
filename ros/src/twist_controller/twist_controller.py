from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import math

import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
TIMEOUT = 2.0

class Controller(object):
    def __init__(self, data):
        self.mass = data['vehicle_mass'] + data['fuel_capacity'] * GAS_DENSITY
        self.brake_deadband = data['brake_deadband']
        self.wheel_radius = data['wheel_radius']
        self.steer_ratio = data['steer_ratio']
        self.max_steer = abs(data['max_steer_angle'])
        self.last_t = rospy.rostime.get_time()
        self.decel = abs(data['decel_limit'])
        self.accel = data['accel_limit']

        # Define Filters
        self.throttle_PID = PID(0.1, 0.1, 0.005) # -data['decel_limit'], data['accel_limit'])
        
        self.steer_control = YawController(data['wheel_base'], data['steer_ratio'], ONE_MPH, \
                                            data['max_lat_accel'], self.max_steer)
                                    
        self.LPF = LowPassFilter(0.2, 1.0)

    def control(self, target_v, target_yaw, current_v, enabled):

        if self.last_t is None:
            return 0.0, 0.0, 0.0

        delta_t = rospy.rostime.get_time() - self.last_t
        self.last_t = rospy.rostime.get_time()

        if not enabled:
            self.throttle_PID.reset()
            return 0.0, 0.0, 0.0

        throttle, brake, steer = 0, 0, 0

        diff_v = target_v - current_v
        accel = self.throttle_PID.step(diff_v, delta_t)
        rospy.logwarn("{} {} {}".format(target_v, diff_v, accel))

        if diff_v < 0:
            brake = self.get_torque(abs(accel), self.mass, self.wheel_radius)
        else:
            throttle = min(self.accel, accel)
            # cte = max(self.decel, min(self.accel, diff_v / delta_t))
            # throttle = max(0.0, min(1.0, throttle))

        if target_v < 0.3:
            brake = self.get_torque(self.decel, self.mass, self.wheel_radius)

        if abs(diff_v) < 0.2:
            self.throttle_PID.reset()

        steer = self.steer_control.get_steering(target_v * ONE_MPH, target_yaw, current_v * ONE_MPH)
        steer = self.LPF.filt(steer)

        # rospy.logwarn("{} {} <{} {} {}>".format(target_v, current_v, throttle, brake, steer))
        return throttle, brake, steer

    # Return brake torke [N*m]
    def get_torque(self, accel, weight, radius):
        return accel * weight * radius
