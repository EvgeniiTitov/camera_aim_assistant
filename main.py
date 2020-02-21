from model import YOLOv3
from model import ComponentsDetector, PolesDetector
from utils import ResultsManager
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

    def assist(self, cap):
        """
        Needs to be a generator?
        :return:
        """

        if not cap.isOpened():
            raise IOError("Failed to open the cap")

        while 1:

            # TODO: Might take a lot of time, consider threading to decode frames in advance
            try:
                has_frame, frame = cap.read()
            except:
                raise IOError("Failed to decode a frame")

            poles = self.pole_detector.predict(image=frame)

            components = self.components_detector.predict(image=frame,
                                                          pole_predictions=poles)

            # Might not need to do it. Just send components
            detected_objects = {**poles, **components}

            if detected_objects:
                # Won't be drawing BBs. Remove after testing
                self.results_manager.draw_bbs(objects_detected=detected_objects,
                                              image=frame)


            cv2.imshow("frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    # video
    #path_to_data = ''

    # image
    #path_to_data = ''

    # webcam
    path_to_data = 0

    video_capture = cv2.VideoCapture(path_to_data)

    AimAssistant().assist(video_capture)

    # TODO: Interfaces. What triggers it and how
    # TODO: What it returns. It needs to be a generator right? To get predictions for each frame
    # TODO: Actual angle calculating algorithm
    # TODO: What to do if multiple objects detected? Camera needs to do it one by one? (Hovers in place?), mark chosen object? (tracking, too complicated)