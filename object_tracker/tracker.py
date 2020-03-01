from collections import OrderedDict
import numpy as np
from scipy.spatial import distance as scipy_dist


class CentroidTracker(object):
    """
    OrderedDict - remembers the order entries were added
    """
    def __init__(
            self,
            max_disappeared=50,
            max_distance=50
    ):
        # Counter to assign unique IDs to each object we want to track
        self.next_object_ID = 0

        # All objects that are being tracked. object ID : centroid centre coordinates
        self.tracked_objects = OrderedDict()

        # Dictionary to keep track how long the object getting tracked have been
        # lost for - N of consecutive frames a particular object ID's been marked as lost
        self.disappeared = OrderedDict()

        # Max N of frames an object can be marked as disappeared until it gets
        # removed from the dictionary of the tracked objects
        self.max_disappeared = max_disappeared

        # Max L btwn centroids to associate an object - the L > than max L allowed,
        # we will start marking the object as disappeared
        self.max_distance = max_distance

    def register(
            self,
            centroid: tuple
    ) -> None:
        """
        Adds new object(s) to the tracker.
        :param centroid:
        :return:
        """
        # Add new element to the dict of tracked objects.
        self.tracked_objects[self.next_object_ID] = centroid

        # Set its disappear counter to 0
        self.disappeared[self.next_object_ID] = 0

        # Increment ID, so the next object coming in will get +1
        self.next_object_ID += 1

    def deregister(
            self,
            object_id: int
    ) -> None:
        """
        Remove long lost (max_disap frames) objects from the tracking dicts
        :param object_id:
        :return:
        """
        del self.tracked_objects[object_id]
        del self.disappeared[object_id]

    def update(
            self,
            bbs: list
    ) -> dict:
        """
        List of coordinates of objects on a new frame obtained either by
        (1) inference or (2) object tracking algorithm.
        Associates new objects to the currently getting tracked ones.
        :param bbs: List of tuples (startX, startY, endX, endY)
        :return: Dict of objects getting tracked
        """
        # If no objects found on a frame, mark all currently being tracked
        # objects as missing, and remove long lost ones > max N disappeard
        if len(bbs) == 0:

            # Loop over the existing objects, mark as disappeared.
            # d.keys() doesnt return a list! Its a class object
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                # Check if any objects need to go
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            return self.tracked_objects

        # else, some objects were detected on the frame
        # initialize an array of input centroids for the new frame
        input_centroids = np.zeros((len(bbs), 2), dtype="int")

        # Loop over the detected objects
        for i, (start_X, start_Y, end_X, end_Y) in enumerate(bbs):
            # Find each object centroid's centre
            centre_X = int((start_X + end_X) / 2.0)
            centre_Y = int((start_Y + end_Y) / 2.0)
            input_centroids[i] = (centre_X, centre_Y)

        # If not tracking anything, just started, no point trying to associate
        # new objects, just register all new centroids
        if len(self.tracked_objects) == 0:

            for i in range(0, len(input_centroids)):
                # Send centroid coordinates for registration. Each will be given new ID
                # and stored in self.tracked_objects
                self.register(input_centroids[i])
        else:
            # else -> update any existing object (x,y)-coordinates based on the centroid
            # location that minimizes the Euclidean distance between them
            object_IDs = list(self.tracked_objects.keys())
            object_centroids = list(self.tracked_objects.values())

            #https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/

            # We need to match an input new centroids to an existing object centroids
            # Compute the distance btwn each pair of new centroids and already being tracked ones
            # output - np.array shape of our distance map (# of tracked centroids, # of input centroids)
            dist = scipy_dist.cdist(np.array(object_centroids), input_centroids)  # Needs np.ndarrays

            # To do matching we need: 1. Find the smallest value in each row and then 2. sort the row
            # indeces based on their min values so that the row with the smallest value is at the front
            # of the index list
            rows = dist.min(axis=1).argsort()

            # Same process - find smallest value in each column and sort using the previously computed
            # row index list
            columns = dist.argmin(axis=1)[rows]

            # Use distances to try to associate object IDs
            used_rows, used_columns = set(), set()

            for row, col in zip(rows, columns):

                # Check if we've already checked the row or col value already
                if row in used_rows or col in used_columns:
                    continue

                # Check the distance between centroids. If its >, than we assume these are
                # not the same object, do not associate them
                if dist[row, col] > self.max_distance:
                    continue

                # Found an input centroid that 1. has the smallest Euclidean distance to an existing centroid
                # 2. hasn't been matched yet
                # Grab the object ID for the current row, set its new centroid and reset the
                # disappeared counter
                object_id = object_IDs[row]
                self.tracked_objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                # Mark row and col as checked
                used_rows.add(row)
                used_columns.add(col)

            # Compute both rows and cols index we haven't yet examined.
            unused_rows = set(range(0, dist.shape[0])).difference(used_rows)
            unused_columns = set(range(0, dist.shape[1])).difference(used_columns)

            # If N of tracked centroids >= N of input centroids, check if any of these objects have
            # potentially disappeared (came more than we used to track) and mark accordingly
            if dist.shape[0] >= dist.shape[1]:
                # Loop over the unused row indexes
                for row in unused_rows:

                    object_id = object_IDs[row]
                    self.disappeared[object_id] += 1

                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)

            else:
                # N of input centroids > N of existing centroids, register new objects
                for col in unused_columns:
                    self.register(input_centroids[col])

        return self.tracked_objects
