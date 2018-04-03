from styx_msgs.msg import TrafficLight
import numpy as np
import cv2
import os
import time

import tensorflow as tf

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.count = 0

        PATH = os.path.dirname(os.path.abspath(__file__)) + '/frozen_inference_graph.pb'

        # self.path = os.path.abspath(__file__)

        #self.saver = tf.train.import_meta_graph(os.path.dirname(self.path + '/model.ckpt.meta'))

        #self.graph = self.load_graph()
        self.detection_graph = tf.Graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(PATH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph, config=config)

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')


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

        image_np = np.expand_dims(image, axis=0)
        #print(image_np.shape)

        t0 = time.time()
        with self.detection_graph.as_default():
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict = {self.image_tensor: image_np}
            )
        t1 = time.time()

        print("[TLClassifier::get_classification] Prediction time: {0:4.4f}ms".format(1000 * (t1-t0)))

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        counts = np.zeros((5))
        for i,score in enumerate(scores):
            if score < 0.8:
                continue
            
            if int(classes[i]) == 1: # Predicted GREEN
                counts[2] += 1 # Update Green bin (specified in styx_msgs/TrafficLight)
            elif int(classes[i]) == 2: # Predicted RED
                counts[0] += 1 # Update Red bin (specified in styx_msgs/TrafficLight)
            elif int(classes[i]) == 3: # Predicted YELLOW
                counts[1] += 1 # Update YELLOW bin (specified in styx_msgs/TrafficLight)
            else: # Predicted None or Off
                counts[4] += 1 # Update None bin (specified in styx_msgs/TrafficLight)

        output = counts.argmax()

        if counts[output] == 0:
            return TrafficLight.UNKNOWN

        if output == 0:
            print("[TLClassifier::get_classification] Prediction: RED")
            return TrafficLight.RED
        elif output == 1:
            print("[TLClassifier::get_classification] Prediction: YELLOW")
            return TrafficLight.YELLOW
        elif output == 2:
            print("[TLClassifier::get_classification] Prediction: GREEN")
            return TrafficLight.GREEN

        return TrafficLight.UNKNOWN
