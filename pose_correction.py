import os

# Speeding up the cv2 video capture
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import logging
import mediapipe as mp
import numpy as np
from typing import List, Mapping, Optional, Tuple, Union
import dataclasses
import cv2
import math
import numpy as np
from mediapipe.framework.formats import landmark_pb2

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_BGR_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# creating the logger object
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


@dataclasses.dataclass
class DrawingSpec:
    # Color for drawing the annotation. Default to the white color.
    color: Tuple[int, int, int] = WHITE_COLOR
    # Thickness for drawing the annotation. Default to 2 pixels.
    thickness: int = 2
    # Circle radius. Default to 2 pixels.
    circle_radius: int = 2
    error_color: Tuple[int, int, int] = RED_COLOR


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (
            value < 1 or math.isclose(1, value)
        )

    if not (
        is_valid_normalized_value(normalized_x)
        and is_valid_normalized_value(normalized_y)
    ):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def draw_landmarks(
    image: np.ndarray,
    landmark_list: landmark_pb2.NormalizedLandmarkList,
    world_landmarks,
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: Optional[
        Union[DrawingSpec, Mapping[int, DrawingSpec]]
    ] = DrawingSpec(color=RED_COLOR),
    connection_drawing_spec: Union[
        DrawingSpec, Mapping[Tuple[int, int], DrawingSpec]
    ] = DrawingSpec(),
    is_drawing_landmarks: bool = True,
):
    """Draws the landmarks and the connections on the image (Modified to draw in RED for wrong poses).

    Args:
      image: A three channel BGR image represented as numpy ndarray.
      landmark_list: A normalized landmark list proto message to be annotated on
        the image.
      connections: A list of landmark index tuples that specifies how landmarks to
        be connected in the drawing.
      landmark_drawing_spec: Either a DrawingSpec object or a mapping from hand
        landmarks to the DrawingSpecs that specifies the landmarks' drawing
        settings such as color, line thickness, and circle radius. If this
        argument is explicitly set to None, no landmarks will be drawn.
      connection_drawing_spec: Either a DrawingSpec object or a mapping from hand
        connections to the DrawingSpecs that specifies the connections' drawing
        settings such as color and line thickness. If this argument is explicitly
        set to None, no landmark connections will be drawn.
      is_drawing_landmarks: Whether to draw landmarks. If set false, skip drawing
        landmarks, only contours will be drawed.

    Raises:
      ValueError: If one of the followings:
        a) If the input image is not three channel BGR.
        b) If any connetions contain invalid landmark index.
    """
    if not landmark_list:
        return
    if image.shape[2] != _BGR_CHANNELS:
        raise ValueError("Input image must contain three channel bgr data.")
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    error_flag = 0
    right_elbow, right_shoulder = 0, 0
    shoulder_bool = False
    for idx, landmark in enumerate(landmark_list.landmark):
        if (
            landmark.HasField("visibility")
            and landmark.visibility < _VISIBILITY_THRESHOLD
        ) or (
            landmark.HasField("presence") and landmark.presence < _PRESENCE_THRESHOLD
        ):
            continue
        landmark_px = _normalized_to_pixel_coordinates(
            landmark.x, landmark.y, image_cols, image_rows
        )
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px

        if idx == 12:
            right_shoulder = landmark.y
            shoulder_bool = True
        if idx == 14 and shoulder_bool:
            right_elbow = landmark.y

    if abs(right_elbow) < abs(right_shoulder):
        error_flag = 1

    if connections:
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(
                    f"Landmark index is out of range. Invalid connection "
                    f"from landmark #{start_idx} to landmark #{end_idx}."
                )
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                drawing_spec = (
                    connection_drawing_spec[connection]
                    if isinstance(connection_drawing_spec, Mapping)
                    else connection_drawing_spec
                )
                if error_flag and start_idx == 12 and end_idx == 14:
                    logging.info("Error: Drawing red line.")
                    cv2.line(
                        image,
                        idx_to_coordinates[start_idx],
                        idx_to_coordinates[end_idx],
                        drawing_spec.error_color,
                        drawing_spec.thickness,
                    )
                else:
                    cv2.line(
                        image,
                        idx_to_coordinates[start_idx],
                        idx_to_coordinates[end_idx],
                        drawing_spec.color,
                        drawing_spec.thickness,
                    )
    # Draws landmark points after finishing the connection lines, which is
    # aesthetically better.
    if is_drawing_landmarks and landmark_drawing_spec:
        for idx, landmark_px in idx_to_coordinates.items():
            drawing_spec = (
                landmark_drawing_spec[idx]
                if isinstance(landmark_drawing_spec, Mapping)
                else landmark_drawing_spec
            )
            # White circle border
            circle_border_radius = max(
                drawing_spec.circle_radius + 1, int(drawing_spec.circle_radius * 1.2)
            )
            cv2.circle(
                image,
                landmark_px,
                circle_border_radius,
                WHITE_COLOR,
                drawing_spec.thickness,
            )
            # Fill color into the circle
            cv2.circle(
                image,
                landmark_px,
                drawing_spec.circle_radius,
                drawing_spec.color,
                drawing_spec.thickness,
            )
        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)


def detect_from_webcam():
    logging.info("Video capture starting.....")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    logging.info("Video capture started")
    with mp_holistic.Holistic(
        model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                logger.warning("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
            )
            draw_landmarks(
                image,
                results.pose_landmarks,
                results.pose_world_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )

            cv2.imshow("MediaPipe Holistic (Press q to exit)", image)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break
    cap.release()


if __name__ == "__main__":
    detect_from_webcam()
