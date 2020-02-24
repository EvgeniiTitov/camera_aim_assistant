from model import YOLOv3
from model import ComponentsDetector, PolesDetector
import numpy as np
import math


class AimAssistant:
    """
    Helps high resolution camera to find an object it is required to take an image of
    """
    def __init__(self):
        poles_net = YOLOv3()
        components_net = YOLOv3()

        # Initialize predicting networks
        self.pole_detector = PolesDetector(predicting_net=poles_net)
        self.components_detector = ComponentsDetector(predicting_net=components_net)

    def aim(
            self,
            frame: np.ndarray
    ):
        """
        Main method that will be called from another program. Gets a frame, returns a
        dictionary with objects and their essential coordinates
        :param frame:
        :return:
        """
        # Predict poles
        poles = self.pole_detector.predict(image=frame)

        # Predict components on the poles
        components = self.components_detector.predict(image=frame,
                                                      pole_predictions=poles)

        # If any components, calculate angle and write other data as object's attributes
        if components:
            self.calculate_angles(components=components,
                                  frame=frame)
            return components

        else:
            return None

    def calculate_angles(
            self,
            components: dict,
            frame: np.ndarray
    ) -> None:
        """
        :param components: detected components
        :param frame: frame
        :return:
        """
        frame_centre = (frame.shape[1] // 2, frame.shape[0] // 2)  # x,y

        # To keep track of all detected objects
        object_id = 0

        # Loop over key-value pairs (image section : detected items)
        for image_section, elements in components.items():

            # Loop over detected items
            for element in elements:

                # Convert element's coordinates to absolute (now they are relative to the
                # object within which they were detected (if any, else already absolute)
                element_absolute_top = image_section.top + element.BB_top
                element_absolute_bot = image_section.top + element.BB_bottom
                element_absolute_left = image_section.left + element.BB_left
                element_absolute_right = image_section.left + element.BB_right

                # cv2.rectangle(frame,
                #               (element_absolute_left, element_absolute_top),
                #               (element_absolute_right, element_absolute_bot),
                #               (0, 165, 255), 5)

                # Calculate element's BB centre relatively to the whole image
                element_x_centre = (element_absolute_right + element_absolute_left) // 2
                element_y_centre = (element_absolute_bot + element_absolute_top) // 2

                # cv2.circle(frame,
                #            (element_x_centre, element_y_centre),
                #            5, (0, 165, 255), thickness=8)

                # Calculate delta (image centre vs element centre)
                delta_x = abs(frame_centre[0] - element_x_centre)
                delta_y = abs(frame_centre[1] - element_y_centre)

                # Line from image centre to each element
                # cv2.line(frame,
                #          frame_centre,
                #          (element_x_centre, element_y_centre),
                #          (0, 165, 255), thickness=8)

                angle_1 = round(np.rad2deg(np.arctan2(delta_x, delta_y)), 2)
                angle_2 = round(90 - angle_1, 2)

                # Write object's BB centre
                element.BB_centre = (int(element_x_centre), int(element_y_centre))

                # Write object's angles to get captured by the high-res camera
                element.angle_to_get_captured = (angle_1, angle_2)

                # Write object's ID
                element.ID = object_id
                object_id += 1

                # Estimate object's relative diameter
                element_BB_diagonal = math.sqrt((element_absolute_top - element_absolute_bot)**2 +\
                                     (element_absolute_right - element_absolute_left)**2)
                frame_diagonal = math.sqrt((frame.shape[0])**2 + (frame.shape[1])**2)

                # TODO: Confirm how to measure object's size relative to the frame
                element.diagonal = element_BB_diagonal / frame_diagonal

        # TODO: Confirm if happy to receive actual class objects with all data in them
        return
