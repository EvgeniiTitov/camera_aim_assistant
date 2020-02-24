from assistant import AimAssistant
from utils import ResultsManager
import cv2
from imutils.video import FPS
import os
import numpy as np

# TODO: Object tracking if decide to take average from 1+ frames

def main():

    aim_assistant = AimAssistant()
    results_manager = ResultsManager()

    video_cap = cv2.VideoCapture(source)

    if not video_cap.isOpened():
        raise IOError("Failed to open the cap")

    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

    output_name = os.path.join(save_path, "video.avi")

    video_writter = cv2.VideoWriter(output_name,
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        10, (round(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             round(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    fps = FPS().start()
    while 1:

        # TODO: Threads for reading/writing results?

        has_frame, frame = video_cap.read()

        if not has_frame:
            break

        predictions = aim_assistant.aim(frame)

        # ! From here you can retrieve all object's data and use for the high-res camera

        if predictions:
            results_manager.draw_bbs(objects_detected=predictions,
                                     image=frame)
            results_manager.check_aim_assistance(components=predictions,
                                                 image=frame)

            # Results extraction check
            for components in predictions.values():
                for comp in components:

                    print(f"Class: {comp.object_name}, Accuracy: {round(comp.confidence, 2)}, "
                          f"ID: {comp.ID}, Diagonal: {round(comp.diagonal, 2)}")


        #cv2.imwrite(os.path.join(save_path, "test.jpg"), frame)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == 27:
            break

        # To write video
        video_writter.write(image=frame.astype(np.uint8))

        fps.update()

    fps.stop()
    print("FPS: {:.2f}".format(fps.fps()))

    video_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    save_path = r'D:\Desktop\system_output\aim_assistance'

    #source = 0
    source = r'D:\Desktop\Reserve_NNs\Datasets\IMAGES_ROW_DS\videos_Oleg\Some_Videos\1.mp4'

    #source = r'D:\Desktop\system_output\aim_assistance\test2.jpg'
    #source = r'D:\Desktop\system_output\TEST_IMAGES\28.JPG'
    #source = r'D:\Desktop\system_output\TEST_IMAGES\DSCN2863.JPG'

    main()
