# import the necessary packages
import argparse
import os
import sys
from collections import deque

import cv2
import numpy as np
import pyrealsense2 as rs
from calibrate import IMG_HEIGHT, IMG_WIDTH, REFERENCE_ID, load_coefficients
from imutils.video import VideoStream

MARKER_SIZE = 0.04
ROBOT_TRANSLATION_1 = np.array(
    [0.39397444, -0.01577215, -0.00178543], dtype=np.float64
)  # this is output by the calibration script
ROBOT_TRANSLATION_2 = np.array(
    [0.4208286, -0.00772413, 0.00511138], dtype=np.float64
)  # this is output by the calibration script
OFFSET = np.array([0.0, -0.0, 0.0])
CUBE_IDS = [26, 27, 28, 29, 30, 31]
SRs = ["realsenseSR_1", "realsenseSR_2"]

DATA_PATH = "/tmp/cube_estimator.txt"


# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

# https://stackoverflow.com/questions/46363618/aruco-markers-with-opencv-get-the-3d-corner-coordinates/46370215#46370215


def inversePerspective(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R = np.matrix(R).T
    invTvec = np.dot(R, np.matrix(-tvec))
    invRvec, _ = cv2.Rodrigues(R)
    return invRvec, invTvec


def relativePosition(rvec1, tvec1, rvec2, tvec2):
    """Get relative position for rvec2 & tvec2. Compose the returned rvec & tvec to use composeRT with rvec2 & tvec2"""
    rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape((3, 1))
    rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))
    # Inverse the second marker
    invRvec, invTvec = inversePerspective(rvec2, tvec2)
    info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
    composedRvec, composedTvec = info[0], info[1]
    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))
    return composedRvec, composedTvec


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str, default="DICT_4X4_50", help="type of ArUCo tag to detect")
ap.add_argument("-c", "--calibration", type=str, default="calibration.yaml")
args = vars(ap.parse_args())

reference_rvec_1, reference_tvec_1, reference_rvec_2, reference_tvec_2 = load_coefficients(args["calibration"])
position_buffer = deque(maxlen=6)

if os.path.exists(DATA_PATH):
    os.remove(DATA_PATH)
f_dump = open(DATA_PATH, "a")

# verify that the supplied ArUCo tag exists and is supported by
# OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported".format(args["type"]))
    sys.exit(0)

# load the ArUCo dictionary and grab the ArUCo parameters
print("[INFO] detecting '{}' tags...".format(args["type"]))
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()
# initialize the video stream and allow the camera sensor to warm up

print("[INFO] starting video stream...")


pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device(SRs[0])
config_1.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, 30)
pipeline_1.start(config_1)
frames_1 = pipeline_1.wait_for_frames()
frame_1 = frames_1.get_color_frame()
color_intrinsics_1 = frame_1.profile.as_video_stream_profile().intrinsics
matrix_coefficients_1 = np.array(
    [[color_intrinsics_1.fx, 0, color_intrinsics_1.ppx], [0, color_intrinsics_1.fy, color_intrinsics_1.ppy], [0, 0, 1]]
)
distortion_coefficients_1 = np.asarray(color_intrinsics_1.coeffs)

pipeline_2 = rs.pipeline()
config_2 = rs.config()
config_2.enable_device(SRs[0])
config_2.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, 30)
pipeline_2.start(config_2)
frames_2 = pipeline_2.wait_for_frames()
frame_2 = frames_2.get_color_frame()
color_intrinsics_2 = frame_2.profile.as_video_stream_profile().intrinsics
matrix_coefficients_2 = np.array(
    [[color_intrinsics_2.fx, 0, color_intrinsics_2.ppx], [0, color_intrinsics_2.fy, color_intrinsics_2.ppy], [0, 0, 1]]
)
distortion_coefficients_2 = np.asarray(color_intrinsics_2.coeffs)

# loop over the frames from the video stream
idx = 0

while True:
    if idx % 2 == 0:
        pipeline = pipeline_1
        matrix_coefficients = matrix_coefficients_1
        distortion_coefficients = distortion_coefficients_1
        reference_rvec = reference_rvec_1
        reference_tvec = reference_tvec_1
        translation = ROBOT_TRANSLATION_1
    else:
        pipeline = pipeline_2
        matrix_coefficients = matrix_coefficients_2
        distortion_coefficients = distortion_coefficients_2
        reference_rvec = reference_rvec_2
        reference_tvec = reference_tvec_2
        translation = ROBOT_TRANSLATION_2
    idx += 1

    frames = pipeline.wait_for_frames()
    frame = frames.get_color_frame()
    if not frame:
        continue
    frame = np.asarray(frame.get_data())

    # detect ArUco markers in the input frame
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

    cv2.aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers

    if len(corners) > 0:
        ids = ids.flatten()
        center_estimates = []
        for markerCorner, markerID in zip(corners, ids):
            if markerID == REFERENCE_ID:
                continue
            if markerID not in CUBE_IDS:
                continue
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                markerCorner, MARKER_SIZE, matrix_coefficients, distortion_coefficients
            )
            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, MARKER_SIZE / 2)
            # Now we have the vectors needed
            composedRvec, composedTvec = relativePosition(rvec, tvec, reference_rvec, reference_tvec)
            # Now compute the actual center of the marker cube
            cube_center = np.array([[0, 0, -MARKER_SIZE / 2]]).T
            rot_mat, _ = cv2.Rodrigues(composedRvec)
            cube_center = rot_mat.dot(cube_center) + composedTvec
            center_estimates.append(cube_center)

        if len(center_estimates) > 0:
            predicted_center = np.mean(np.array(center_estimates), axis=0).squeeze(-1)
            predicted_center = predicted_center + translation
            predicted_center = predicted_center + OFFSET
            position_buffer.append(predicted_center)
            pred = np.mean(position_buffer, axis=0)
            print(pred)
            # Write the buffer to the socket
            f_dump.write(str(pred) + "\n")
            f_dump.flush()

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
f_dump.close()
pipeline_1.stop()
pipeline_2.stop()
