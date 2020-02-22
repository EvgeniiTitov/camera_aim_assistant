from model import YOLOv3
from model import ComponentsDetector, PolesDetector
from utils import ResultsManager, calculate_angles
import cv2


class AimAssistant:
    """
    Helps high resolution camera to find an object it is required to take an image of
    """
    def __init__(self):
        poles_net = YOLOv3()
        components_net = YOLOv3()

        self.pole_detector = PolesDetector(predicting_net=poles_net)
        self.components_detector = ComponentsDetector(predicting_net=components_net)

        self.results_manager = ResultsManager()

    def assist(
            self,
            cap=None,
            image=None
    ):
        """
        Needs to be a generator?
        :return:
        """
        if image is None:
            if not cap.isOpened():
                raise IOError("Failed to open the cap")

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

        while 1:

            # TODO: Might take a lot of time, consider threading to decode frames in advance
            if cap:
                try:
                    has_frame, frame = cap.read()
                except:
                    raise IOError("Failed to decode a frame")
            else:
                frame = image

            poles = self.pole_detector.predict(image=frame)

            components = self.components_detector.predict(image=frame,
                                                          pole_predictions=poles)

            # Might not need to do it. Just send components, don't care about poles
            detected_objects = {**poles, **components}

            if detected_objects:
                # Send all objects to calculate their relative angles saved as attibutes
                calculate_angles(components=components,
                                 frame=frame)

                # Won't be drawing BBs. Remove after testing
                #self.results_manager.draw_bbs(objects_detected=detected_objects,
                #                              image=frame)

                self.results_manager.check_aim_assistance(components=components,
                                                          image=frame)

                # TODO: It needs to be a generator, returns angles for each frame? Angles need to be sent
                # TODO: to the mission's file?! (Anton, Lena)

                # TODO: Do we need it process just one frame from the correct position? OR multiple?

            #cv2.imshow("frame", frame)
            cv2.imwrite(f"D:\Desktop\system_output/aim_assistance\img.jpg", frame)
            #cv2.waitKey(0)

            if image is not None:
                break

        #cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    # video: Try FPV video you have
    #path_to_data = ''

    # image
    path_to_data = r'D:\Desktop\system_output\TEST_IMAGES\28.jpg'
    image = cv2.imread(path_to_data)

    # webcam
    #path_to_data = 0
    #video_capture = cv2.VideoCapture(path_to_data)

    AimAssistant().assist(image=image)

    # TODO: Interfaces. What triggers it and how
    # TODO: Actual angle calculating algorithm