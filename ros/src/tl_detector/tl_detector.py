#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math

STATE_COUNT_THRESHOLD = 3
MIN_LIGHT_DISTANCE = 100

# DEBUG: Get images to train the classifier
import glob
import os
import random
GET_IMAGES = False
RED_FOLDER = os.getcwd() + '/images/red/'
YELLOW_FOLDER = os.getcwd() + '/images/yellow/'
GREEN_FOLDER = os.getcwd() + '/images/green/'
IMG_INTERVAL = 0.5

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.light_classifier = TLClassifier()

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.listener = tf.TransformListener()

        self.Image_Save_Init()
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''

        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state

            if state == TrafficLight.RED:
                light_wp = light_wp
            elif state == 1 or state == 2:
                light_wp = -2
            else:
                light_wp = -1
                
            #light_wp = light_wp if state == TrafficLight.RED else -1
            
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        if self.waypoints is None or len(self.waypoints) == 0:
            return -1

        min_dst = float("inf")
        index = -1

        for i, wp in enumerate(self.waypoints):
            dst = self.module(pose.position, wp.pose.pose.position)

            if dst < min_dst:
                min_dst = dst
                index = i

        # If the waypoint is behind the car, get the next one
        wp = self.waypoints[index]
        angle = self.get_yaw(pose.orientation) - math.atan2(wp.pose.pose.position.y - pose.position.y, wp.pose.pose.position.x - pose.position.x)

        if abs(angle) > math.pi / 4:
            index = (index + 1) % len(self.waypoints)
            wp = self.waypoints[index]

        return index
    
    def get_yaw(self,orientation):
        angles = tf.transformations.euler_from_quaternion((orientation.x, orientation.y, orientation.z, orientation.w))
        return angles[2]

    def module(self, pos1, pos2):
        return math.sqrt( (pos1.x - pos2.x)*(pos1.x - pos2.x) + (pos1.y - pos2.y)*(pos1.y - pos2.y))

    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        if GET_IMAGES:
            self.Image_Save(light.state, cv_image)
                    
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)

        stop_lines = self.config['stop_line_positions']

        if self.lights:
            # Find the closest trafic light in front of the car
            min_dst = float("inf")
            index = -1
            for i in range(len(stop_lines)):
                dst = self.distance(stop_lines[i][0], stop_lines[i][1], \
                                    self.pose.pose.position.x, self.pose.pose.position.y)

                if dst < min_dst:
                    index = i
                    min_dst = dst

            angle = self.get_yaw(self.pose.pose.orientation) - math.atan2(stop_lines[index][1] - self.pose.pose.position.y, stop_lines[index][0] - self.pose.pose.position.x)

            if abs(angle) > math.pi / 4:
                index = (index + 1) % len(stop_lines)

            dst = self.distance(stop_lines[index][0], stop_lines[index][1], \
                    self.pose.pose.position.x, self.pose.pose.position.y)
            
            # If the traffic light is not close, return unknown state (reduce computation)
            if dst > MIN_LIGHT_DISTANCE:
                return -1, TrafficLight.UNKNOWN
            
            # Define the Pose for the closest traffic light in front of the car
            # and find its closest waypoint
            stop_light = Pose()
            stop_light.position.x = stop_lines[index][0]
            stop_light.position.y = stop_lines[index][1]
            stop_light.position.z = 0

            light_wp = self.get_closest_waypoint(stop_light)
            light = self.lights[index]

        # Return the traffic light state if we found one
        # else return unknown state
        if light:
            state = self.get_light_state(light)
            #rospy.logwarn("[tl_detector::image_cb] Light<{}>: {} - {}".format(index, str(state), light.state))
            return light_wp, state  # USE PREDICTED STATE
            #return light_wp, light.state # USE STATE FROM SIMULATOR DATA

        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

    def Image_Save_Init(self):
        if not GET_IMAGES:
            return

        folder = RED_FOLDER
        if not os.path.isdir(folder):
            os.makedirs(folder)

        folder = YELLOW_FOLDER
        if not os.path.isdir(folder):
            os.makedirs(folder)

        folder = GREEN_FOLDER
        if not os.path.isdir(folder):
            os.makedirs(folder)

        self.image_timer = rospy.get_time()

    def Image_Save(self, state, cv_image):
        rospy.logwarn("[Image_Save]")
        if rospy.get_time() - self.image_timer < IMG_INTERVAL:
            return

        folder = ''
        if state == 0:
            folder = RED_FOLDER
        elif state == 1:
            folder = YELLOW_FOLDER
        elif state == 2:
            folder = GREEN_FOLDER
        else:
            return

        self.image_timer = rospy.get_time()

        if state == 0 and random.random() > 0.1:
            return

        index = len(os.listdir(folder))
        filename = folder + 'img_b_' + str(index) + '.jpg'

        rospy.logwarn("SAVED IMAGE")
        cv2.imwrite(filename, cv_image)

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
