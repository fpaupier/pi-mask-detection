from typing import Tuple

from PIL import Image
from datetime import datetime


class Alert:
    """
    Alert is an object containing data and metadata about an event "someone not wearing a mask".
    """

    def __init__(self,
                 device: str,
                 img: Image,
                 prob: float,
                 location: Tuple[float, float],
                 utc_ts: datetime,
                 face_detection_model: str,
                 mask_classifier_model: str):
        """
        All timestamps are expressed in UTC.
        location information are passed in decimal degrees under a (latitude, longitude) tuple.

        Args:
            device: the device which created the event
            img:
            prob:
            location: (lat, long)
            utc_ts: the event time
            face_detection_model: reference to the model used to detect face e.g. "ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"
            mask_classifier_model: reference to the classifier model mask/no mask
        """
        self.created_at: datetime = datetime.utcnow()
        self.created_by: str = device
        self.image: Image = img
        self.probability: float = prob
        self.event_time: datetime = utc_ts
        self.face_detection_model = face_detection_model
        self.mask_classifier_model = mask_classifier_model
        self.location: Tuple[float, float] = location

    def __str__(self):
        return f" === Alert ===\n" \
               f"- Created at: {self.created_at}\n" \
               f"- Event time: {self.event_time}\n" \
               f"- Created by: {self.created_by}\n" \
               f"- Latitude, longitude: {self.location}\n" \
               f"- Image size: {self.image.size}\n" \
               f"- Probability of event: {self.probability}\n" \
               f"- Face detection model used: {self.face_detection_model}\n" \
               f"- Mask classifier model used: {self.mask_classifier_model}\n" \
