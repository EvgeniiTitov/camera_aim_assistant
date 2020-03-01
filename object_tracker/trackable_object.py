class TrackableObject(object):
    """
    Represents an object that gets tracked
    """
    def __init__(
            self,
            object_id,
            centroid
    ):
        # Initialize a list of centroids using the current
        # one provided during class initialization
        self.object_id = object_id
        self.centroids = [centroid]

        # Flag whether an object's already been counted
        self.been_counted = False
