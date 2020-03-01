from model import YOLOv3
from model import ComponentsDetector, PolesDetector
from utils import ResultsManager
from object_tracker import CentroidTracker
from imutils.video import FPS
import argparse
import multiprocessing
import numpy as np
import imutils
import cv2
import dlib
import os
import sys

"""
We reinstantiate trackers every N frames, but they do not do actual object 
tracking in a sense they do not tell if an object on frame N is the same as 
the object on the frame N + 10. This tracking is done by CentroidTracker
"""

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_path", type=str, help="Path to save processed video stream")
    parser.add_argument("--source", default="0",
                        help="video source to process: video path or webcam (0)")
    parser.add_argument("--skip", default=15, type=int, help="How often a NN is run")

    return parser.parse_args()


def create_tracker(
        bb: list,
        label: str,
        RGB_frame: np.ndarray,
        input_Q,
        output_Q
) -> None:
    """
    Constructs a dlib rect obj from the BB coordinates and starts the correlation tracker
    :param bb: BB coord of the object to track
    :param label: objects class
    :param RGB_frame: rgb frame
    :param input_Q:
    :param output_Q:
    :return: Puts updated bb into the output Q
    """
    # TODO: Make it report when an object is lost

    # Initialize a tracker
    tracker = dlib.correlation_tracker()
    rect = dlib.rectangle(bb[0], bb[1], bb[2], bb[3])
    tracker.start_track(RGB_frame, rect)

    # daemon process
    while True:
        # Grab a frame from the input Q
        rgb_frame = input_Q.get()

        # If there was a frame, process it
        if rgb_frame is not None:

            # Update the tracker and grab the position of the tracked object
            tracker.update(rgb_frame)
            position = tracker.get_position()
            startX = int(position.left())
            startY = int(position.top())
            endX = int(position.right())
            endY = int(position.bottom())

            # Add the label and BB coordinates to the output Q
            output_Q.put((label, (startX, startY, endX, endY)))


def main():
    centroid_tracker = CentroidTracker(max_disappeared=40,
                                       max_distance=50)

    # Initialize a list to store each dlib correlation tracker and
    # a dict to map each unique object ID to a TrackableObject
    trackers = list()

    writer = None

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError("Failed to open the cap")

    frame_counter = 0

    fps = FPS().start()

    while True:

        has_frame, frame = cap.read()

        if not has_frame:
            break

        # TODO: play with width: changed vs not changed
        # Resize and convert from BGR (OpenCV default) to RGB (dlib's requirement)
        frame = imutils.resize(frame, width=1000)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if writer is None and args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args.save_path, fourcc, 30,
                                     (frame.shape[1], frame.shape[0]), True)

        # List of BB returned by either (1) running a NN or (2) object tracker
        # computing new object's location
        bbs = list()
        status = "Waiting"

        # We want to run nets once in N frames
        if frame_counter % skip_frame == 0:
            status = "Detecting"

            # Initialize a new set of object trackers
            trackers = []

            # Run inference to detect power line poles
            poles = pole_detector.predict(frame)

            # Run inference to detect components
            components = components_detector.predict(image=frame,
                                                     pole_predictions=poles)

            if poles or components:
                results_manager.draw_bbs(objects_detected={**poles, **components},
                                         image=frame)

            # Loop over components, create a separate tracker for each one
            for pole_subimage, comps in components.items():
                for comp in comps:

                    # Objects BB coordinates and class name
                    bb = (pole_subimage.left + comp.BB_left,
                          pole_subimage.top + comp.BB_top,
                          pole_subimage.left + comp.BB_right,
                          pole_subimage.top + comp.BB_bottom)

                    obj_class = comp.object_name

                    # Create new tracker for each detected component. Will be used
                    # during skip frames phase
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(bb[0], bb[1], bb[2], bb[3])
                    tracker.start_track(frame_rgb, rect)
                    trackers.append((obj_class, tracker))

        # On remaining N-1 frames we do object tracking
        else:
            # Loop over all trackers, make each one compute new object's location
            # given a new frame. Each one outputs results for the object it is bound
            # to.
            for obj_class, tracker in trackers:
                status = "Tracking"

                tracker.update(frame_rgb)

                position = tracker.get_position()
                startX = int(position.left())
                startY = int(position.top())
                endX = int(position.right())
                endY = int(position.bottom())

                # Add updated BB of each object to the main list of bbs
                bbs.append((startX, startY, endX, endY))

                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (255, 0, 0), 2)

                cv2.putText(frame, obj_class, (startX, startY - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)

        # Having got, the new bbs, run the centroid tracker to register/associate/
        # deregister objects detected on the current frame and the old ones
        # Objects is a dict. {object_id: centroid centre, }
        objects = centroid_tracker.update(bbs=bbs)

        cv2.putText(frame, status, (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if writer and args.save_path:
            writer.write(frame)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        frame_counter += 1
        fps.update()

    fps.stop()
    print("Time elapsed: {:.2f}".format(fps.elapsed()))
    print("FPS: {:.2f}".format(fps.fps()))
    print("Total N of objects: {}".format(centroid_tracker.next_object_ID))

    if writer:
        writer.release()

    cap.release()
    cv2.destroyAllWindows()


# def main_multiprocessing():
#
#     # Qs for communication to other processes
#     input_Qs = list()
#     output_Qs = list()
#
#     writer = None
#     cap = cv2.VideoCapture(source)
#     if not cap.isOpened():
#         raise IOError("Failed to open the cap")
#
#     frame_counter = 0
#     fps = FPS().start()
#
#     while True:
#         has_frame, frame = cap.read()
#
#         if not has_frame:
#             break
#
#         # Resize and convert from BGR (OpenCV default) to RGB (dlib's requirement)
#         # TODO: play with width: changed vs not changed
#         frame = imutils.resize(frame, width=800)
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         if writer is None and args.save_path:
#             fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#             writer = cv2.VideoWriter(args.save_path, fourcc, 30,
#                                      (frame.shape[1], frame.shape[0]), True)
#
#         # Run inference every N frames. Track objects N - 1 frames. frame_counter % 5 == 0
#         if len(input_Qs) == 0:
#             # IF YOU RUN ONES - IT WORKS
#             # TODO: Clean your Qs, delete processes. Or, it will track the same object 1+ times
#
#             print("Inference on frame: {}".format(frame_counter))
#
#             # TODO: Try sending not RGB frame, anything changes?
#             # TODO: Should we also track the poles? Update its coordinates less often
#
#             # Run inference to detect power line poles
#             poles = pole_detector.predict(frame)
#             # Run inference to detect components
#             components = components_detector.predict(image=frame,
#                                                      pole_predictions=poles)
#
#             # Draw poles BB if any
#             if poles:
#                 results_manager.draw_bbs(objects_detected=poles,
#                                          image=frame)
#
#             # If the net failed to find any components, try the next one
#             if not components:
#                 print("No components => *continue*")
#
#                 cv2.imshow("Frame", frame)
#                 key = cv2.waitKey(1) & 0xFF
#
#                 if key == ord("q"):
#                     break
#
#                 fps.update()
#                 frame_counter += 1
#                 continue
#
#             # Loop over components, create a separate tracker in another process
#             # for each component
#             for pole_subimage, comps in components.items():
#                 for comp in comps:
#                     # Objects BB coordinates
#                     bb = (comp.BB_left, comp.BB_top, comp.BB_right, comp.BB_bottom)
#                     print("Objects BB:", bb)
#                     obj_class = comp.object_name
#                     print("Object class:", obj_class)
#
#                     # Create 2 new Qs for an object
#                     input_Q = multiprocessing.Queue()
#                     output_Q = multiprocessing.Queue()
#                     input_Qs.append(input_Q)
#                     output_Qs.append(output_Q)
#
#                     # Spawn a daemon process for a new object tracker. Send there an object
#                     # its class name, the frame and both Qs to communicate both ways
#                     p = multiprocessing.Process(
#                         target=create_tracker,
#                         args=(bb, obj_class, frame_rgb, input_Q, output_Q))
#
#                     # Set its daemon attribute to 1 so that the process keeps running in the
#                     # background until the main one is alive
#                     p.daemon = True
#                     p.start()
#                     print("Started process:", p.name)
#
#                     # Draw the BB, write obj's class name
#                     cv2.rectangle(frame,
#                                   (comp.BB_left, comp.BB_top),
#                                   (comp.BB_right, comp.BB_bottom),
#                                   (0, 255, 0), 2)
#
#                     cv2.putText(frame, obj_class,
#                                 (comp.BB_left, comp.BB_top - 15),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.45,
#                                 (0, 255, 0), 2)
#
#         # Otherwise, we are already detecting objects, so we need to apply each of the
#         # trackers to the new frame
#         else:
#             # TODO: How to drop objects after they've been gone for N frames
#             for Q in input_Qs:
#                 Q.put(frame_rgb)
#
#             # Loop over each of the output Qs
#             for Q in output_Qs:
#                 # grab updated BB for an object. IMPORTANT: .get() is blocking operation,
#                 # so this will pause our main process execution till the respective process
#                 # finishes the tracking update
#                 (label, (startX, startY, endX, endY)) = Q.get()
#
#                 # Draw the BB
#                 cv2.rectangle(frame, (startX, startY), (endX, endY),
#                               (255, 0, 0), 2)
#                 cv2.putText(frame, label, (startX, startY - 15),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
#
#         if writer is not None and args.save_path:
#             writer.write(frame)
#
#         cv2.imshow("Frame", frame)
#         key = cv2.waitKey(1) & 0xFF
#
#         if key == ord("q"):
#             break
#
#         fps.update()
#         frame_counter += 1
#
#     fps.stop()
#     print("FPS: {:.2f}".format(fps.fps()))
#
#     if writer is not None:
#         writer.release()
#
#     cv2.destroyAllWindows()
#     cap.release()


if __name__ == "__main__":
    args = parse_arguments()

    if args.save_path:
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)

    if args.source == "0":
        source = 0
    else:
        if not os.path.isfile(args.source):
            raise IOError("Provided file is not a video")
        source = args.source

    skip_frame = args.skip
    assert 0 <= skip_frame <= 100, "Wrong number of SKIP frames provided"

    # Initialize predicting nets
    poles_net = YOLOv3()
    components_net = YOLOv3()
    pole_detector = PolesDetector(predicting_net=poles_net)
    components_detector = ComponentsDetector(predicting_net=components_net)

    results_manager = ResultsManager()

    main()
    #main_multiprocessing()