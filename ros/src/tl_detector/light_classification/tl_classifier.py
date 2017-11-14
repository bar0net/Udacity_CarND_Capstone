from styx_msgs.msg import TrafficLight
import cv2

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier

        self.count = 0

        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        
        # cv2.imwrite('./'+str(self.count)+'.jpg', image)
        self.count += 1 

        return TrafficLight.UNKNOWN
