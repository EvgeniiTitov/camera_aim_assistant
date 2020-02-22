import cv2
import os
import numpy as np


def calculate_angles(
        components: dict,
        frame: np.ndarray
) -> None:
    """
    :param components: detected components
    :return:
    """
    frame_centre = (frame.shape[1] // 2, frame.shape[0] // 2)  # x,y

    for image_section, elements in components.items():

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

            element.BB_centre = (int(element_x_centre), int(element_y_centre))
            element.angle_to_get_captured = (angle_1, angle_2)

class ResultsManager:
    """
    Class performing BBs drawing, saving objects to disk
    """
    def __init__(
            self,
            save_path=None,
            cropped_path=None
    ):

        self.save_path = save_path
        self.cropped_path = cropped_path

    def line_text_size(self, image):
        """
        Method determining BB line thickness and text size based on the original image's size
        :return:
        """
        line_thickness = int(image.shape[0] * image.shape[1] // 1_000_000)
        text_size = 0.5 + (image.shape[0] * image.shape[1] // 5_000_000)
        text_boldness = 1 + (image.shape[0] * image.shape[1] // 2_000_000)

        return line_thickness, text_size, text_boldness

    def draw_bbs(
            self,
            objects_detected: dict,
            image: np.ndarray
    ) -> None:
        """
        Draws BBs of the objects detected on the image
        :param objects_detected: an iterable containing all objects detected
        :param image: a frame/image on which objects were detected
        :return: None
        """
        colour = (0, 255, 0)

        for image_section, elements in objects_detected.items():

            # There might be multiple objects detected in a certain image section (whole image:poles),
            # pole1:elements, pole2:elements etc.
            for element in elements:

                # Check element class and change BBs colour
                if element.object_name == "insl":
                    colour = (210, 0, 210)
                elif element.object_name == "dump":
                    colour = (255, 0, 0)
                elif element.object_name == "pillar":
                    colour = (0, 128, 255)

                # Draw BBs using both BBs coordinates and coordinates of the image section relative to the original
                # image in which this object was detected (relative coordinates)
                cv2.rectangle(image, (image_section.left + element.BB_left, image_section.top + element.BB_top),
                                     (image_section.left + element.BB_right, image_section.top + element.BB_bottom),
                                      colour, self.line_text_size(image)[0])

                label = "{}:{:1.2f}".format(element.object_name, element.confidence)

                label_size, base_line = cv2.getTextSize(label,
                                                        cv2.FONT_HERSHEY_SIMPLEX,
                                                        self.line_text_size(image)[1], 1)

                top = max(element.top + image_section.top, label_size[1])

                cv2.putText(image, label,
                            (element.left + image_section.left, top),
                            cv2.FONT_HERSHEY_SIMPLEX, self.line_text_size(image)[1],
                            (0, 0, 0), self.line_text_size(image)[-1])

    def check_aim_assistance(
            self,
            components,
            image
    ):
        colour = (0, 165, 255)
        image_centre = (image.shape[1] // 2, image.shape[0] // 2)  # x,y

        # draw axis
        cv2.line(image, (image.shape[1] // 2, 0), (image.shape[1] // 2, image.shape[0]),
                 (3, 248, 252), thickness=3)
        cv2.line(image, (0, image.shape[0] // 2), (image.shape[1], image.shape[0] // 2),
                 (3, 248, 252), thickness=3)

        # point at image centre
        cv2.circle(image, image_centre, 5, colour, thickness=8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        for image_section, elements in components.items():
            for element in elements:
                #cv2.circle(image, element.BB_centre, 6, colour, 8)
                cv2.line(image, image_centre, element.BB_centre, (244, 3, 252), 5)
                cv2.putText(image, str(element.angle_to_get_captured),
                            element.BB_centre, font, 4, (255, 255, 255), 2, cv2.LINE_AA)

        #(str(e) for e in element.angle_to_get_captured)

    def save_objects_detected(
            self,
            image,
            objects_detected,
            video_writer=None,
            frame_counter=None,
            image_name=None
    ):
        """
        Class method saving objects detected (croping them out)
        :param objects_detected:
        :return:
        """
        for image_section, elements in objects_detected.items():

            # Use enumerate() to make sure no objects get overwritten
            for index, element in enumerate(elements, start=1):
                if not video_writer:
                    # Processing image(s)
                    # There used to be image_section.frame[]
                    cropped_frame = image[element.BB_top + image_section.top:
                                          element.BB_bottom + image_section.top,
                                          element.BB_left + image_section.left:
                                          element.BB_right + image_section.left]

                    file_name = image_name + "_" + element.object_name + "_" + str(index) + ".jpg"
                    cv2.imwrite(os.path.join(self.cropped_path, file_name), cropped_frame)
                else:
                    # ! NEEDS TESTING Processing video
                    cropped_frame = image[element.BB_top + image_section.top:
                                          element.BB_bottom + image_section.top,
                                          element.BB_left + image_section.left:
                                          element.BB_right + image_section.left]
                    frame_name = frame_counter + "_" + element.object_name + "_" + str(index) + ".jpg"
                    cv2.imwrite(os.path.join(self.cropped_path, frame_name), cropped_frame)

    def save_frame(
            self,
            image,
            image_name=None,
            video_writer=None
    ):
        """
        Saves a frame with all BBs drawn on it
        :return:
        """
        if video_writer is None:
            image_name = image_name + "_out.jpg"
            cv2.imwrite(os.path.join(self.save_path, image_name), image)
        else:
            video_writer.write(image.astype(np.uint8))

    def draw_the_line(
            self,
            image,
            line,
            tilt_angle
    ):
        """
        Draws a line which is used for a concrete pole tilt defect detection
        :param image:
        :param line:
        :return:
        """
        # height, width
        label = "Angle: {0:.2f}".format(tilt_angle)
        cv2.putText(image, label, (10, int(image.shape[0]/2)), cv2.FONT_HERSHEY_SIMPLEX,
                    self.line_text_size(image)[1],
                    (0, 0, 0), self.line_text_size(image)[-1])
        # Line coordinates are relative to the pillar image, not the original one
        cv2.line(image,
                 (line[0], line[1]),
                 (line[2], line[3]),
                 (0, 0, 255), 4,
                 cv2.LINE_AA)