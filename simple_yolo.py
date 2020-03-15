import cv2
import numpy as np
import math


class NetElements:
    """
   ELEMENTS detecting and classifying neural network
   Receives IMAGE (NumPy array)
   Returns list of the objects (components: INSULATORS, DUMPERS) predicted.
   """

    # ! MIGHT MAKE SENSE TO MOVE THOSE PARAMETERS TO THE TXT FILE SINCE USER DOESN't WANT
    # ! TO DEAL WITH THE CODE BUT WITH A TXT FILE!

    confidence_thresh = 0.15
    NMS_thresh = 0.25
    input_width, input_height = 608, 608

    def __init__(self):
        self.net = self.setup_net()
        print("Initialized")

    def setup_net(self):
        config = r'C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\defect_detection\defect_detection\dependencies/components.cfg'
        weights=r'C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\defect_detection\defect_detection\dependencies/components.weights'


        neural_net = cv2.dnn.readNetFromDarknet(config, weights)
        neural_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        neural_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        return neural_net

    def create_blob(self, image):
        """
       Creates a blob out of the image provided. Returns the blob
       """
        blob = cv2.dnn.blobFromImage(image, 1 / 255,
                                     (self.input_width, self.input_height),
                                     [0, 0, 0], 1, crop=False)

        return blob

    def output_layers(self, net):
        """
       Returns names of the output YOLO layers: ['yolo_82', 'yolo_94', 'yolo_106']
       """
        layers = net.getLayerNames()

        return [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def process_predictions(self, image, predictions):
        """
       Process all BBs predicted. Keep only the valid ones.
       """
        image_height, image_width = image.shape[0], image.shape[1]
        classIds, confidences, boxes = [], [], []
        objects_predicted = list()

        # For each prediction from each of 3 YOLO layers
        for prediction in predictions:

            # For each detection from one YOLO layer
            for detection in prediction:
                scores = detection[5:]
                classId = np.argmax(scores)  # Index of a BB with highest confidence
                confidence = scores[classId]  # Value of this BB's confidence

                if confidence > self.confidence_thresh:
                    # Centre of object relatively to the upper left corner in percent
                    centre_x = int(detection[0] * image_width)
                    centre_y = int(detection[1] * image_height)

                    # Width and height of the BB predicted. Check for ERROR
                    width_percent = detection[2] if detection[2] < 0.98 else 0.98
                    height_percent = detection[3] if detection[3] < 0.98 else 0.98

                    # Calculate actual size of the BB
                    width = int(width_percent * image_width)
                    height = int(height_percent * image_height)

                    # ERROR CATCHING WITH ABS
                    left = int(centre_x - (width / 2)) if int(centre_x - (width / 2)) > 0 else 2
                    top = int(centre_y - (height / 2)) if int(centre_y - (height / 2)) > 0 else 2

                    # Save prediction results
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non-max suppression to eliminate redundant overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_thresh, self.NMS_thresh)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            objects_predicted.append([classIds[i], confidences[i], left,
                                      top, left + width, top + height])

        return objects_predicted

    def predict(self, image):
        """
       Performs utility pole detection and classification. Returns list of objects detected
       """
        blob = self.create_blob(image)

        # Pass the blob to the neural net
        self.net.setInput(blob)

        # Get output YOLO layers from which read predictions
        layers = self.output_layers(self.net)

        # Run forward pass and get predictions from 3 YOLO layers
        predictions = self.net.forward(layers)

        # Parse the predictions, save only the valid ones
        components = self.process_predictions(image, predictions)

        return components


if __name__ == "__main__":

    net = NetElements()

    #source = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080,format=(string)NV12, framerate=(fraction)21/1 ! nvvidconv ! video/x-raw, width=(int)1280, height=int(960), format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'
    video_cap = cv2.VideoCapture(0)
    if not video_cap.isOpened():
        raise IOError("Failed to open the cap")

    #cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    print('Video is about to start')

    while True:
        ret, frame = video_cap.read()

        if not ret:
            break

        predictions = net.predict(frame)
        print("This is prediction", predictions)
        # color=(255,0,0)

        for element in predictions:
            # cv2.rectangle(frame,(prediction[2],prediction[3]),(prediction[4],prediction[5]),color,thickness=2)
            frame_centre = (frame.shape[1] // 2, frame.shape[0] // 2)
            object_id = 0

            # Convert element's coordinates to absolute (now they are relative to the
            # object within which they were detected (if any, else already absolute)
            element_absolute_left = element[2]
            element_absolute_top = element[3]
            element_absolute_right = element[4]
            element_absolute_bot = element[5]


            cv2.rectangle(frame,
                          (element_absolute_left, element_absolute_top),
                          (element_absolute_right, element_absolute_bot),
                          (0, 165, 255), 3)

            # Calculate element's BB centre relatively to the whole image
            element_x_centre = (element_absolute_right + element_absolute_left) // 2
            element_y_centre = (element_absolute_bot + element_absolute_top) // 2

            cv2.circle(frame,
                       (element_x_centre, element_y_centre),
                       5, (0, 165, 255), thickness=4)

            # Calculate delta (image centre vs element centre)
            delta_x = abs(frame_centre[0] - element_x_centre)
            delta_y = abs(frame_centre[1] - element_y_centre)

            # Line from image centre to each element
            cv2.line(frame,
                     frame_centre,
                     (element_x_centre, element_y_centre),
                     (0, 165, 255), thickness=4)

            angle_1 = round(np.rad2deg(np.arctan2(delta_x, delta_y)), 2)
            angle_2 = round(90 - angle_1, 2)

            print(angle_1, angle_2)

            # Write object's BB centre
            #element.BB_centre = (int(element_x_centre), int(element_y_centre))

            # Write object's angles to get captured by the high-res camera
            #element.angle_to_get_captured = (angle_1, angle_2)

            # Write object's ID
            # element.ID = object_id
            # object_id += 1

            # Estimate object's relative diameter
            # element_BB_diagonal = math.sqrt((element_absolute_top - element_absolute_bot) ** 2 + \
            #                                 (element_absolute_right - element_absolute_left) ** 2)
            # frame_diagonal = math.sqrt((frame.shape[0]) ** 2 + (frame.shape[1]) ** 2)
            #
            # # TODO: Confirm how to measure object's size relative to the frame
            #element.diagonal = element_BB_diagonal / frame_diagonal

        cv2.imshow('Frame', frame)

        #print('Video starts')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_cap.release()
    cv2.destroyAllWindows()