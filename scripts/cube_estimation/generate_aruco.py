# Generate 4 AruCo markers.
import argparse
import os
import sys

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output image containing ArUCo tag")
ap.add_argument("-t", "--type", type=str, default="DICT_4X4_50", help="type of ArUCo tag to generate")
ap.add_argument("-n", "--num", type=int, default=10, help="number of tags to generate")
ap.add_argument("-s", "--start", type=int, default=25, help="Index to start generation at")
args = vars(ap.parse_args())

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

# verify that the supplied ArUCo tag exists and is supported by
# OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported".format(args["type"]))
    sys.exit(0)

# load the ArUCo dictionary
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])

if not os.path.exists(args["output"]):
    os.makedirs(args["output"])

for id in range(args["num"]):
    id = id + args["start"]
    # allocate memory for the output ArUCo tag and then draw the ArUCo
    # tag on the output image
    print("[INFO] generating ArUCo tag type '{}' with ID '{}'".format(args["type"], id))
    tag = np.zeros((250, 250, 1), dtype="uint8")
    cv2.aruco.drawMarker(arucoDict, id, 250, tag, 1)
    # write the generated ArUCo tag to disk and then display it to our
    # screen
    cv2.imwrite(os.path.join(args["output"], args["type"] + "_" + str(id) + ".png"), tag)
    cv2.imshow("ArUCo Tag", tag)
    cv2.waitKey(0)
