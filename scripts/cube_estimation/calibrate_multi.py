import argparse

import cv2
import numpy as np
import pyrealsense2 as rs

# https://stackoverflow.com/questions/12299870/computing-x-y-coordinate-3d-from-image-point

IMG_HEIGHT, IMG_WIDTH = 480, 640
MARKER_SIZE = 0.06
CUBE_MARKER_SIZE = 0.04
REFERENCE_ID = 25
DESIRED_ZERO = np.array([0.395, -0.01, 0.025])  # The known 3D position of the reference marker.
CUBE_IDS = [26, 27, 28, 29, 30, 31]  # The IDs of the markers used on the cube
SRs = ["realsenseSR_1", "realsenseSR_2"]  # Modify to be the IDs of your real sense camera


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

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


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


def calibrate_camera(sr, aruco_type):
    print("CALIB")
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
    arucoParams = cv2.aruco.DetectorParameters_create()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(sr)
    config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, 30)
    pipeline.start(config)
    frames = pipeline.wait_for_frames()
    frame = frames.get_color_frame()
    color_intrinsics = frame.profile.as_video_stream_profile().intrinsics
    matrix_coefficients = np.array(
        [[color_intrinsics.fx, 0, color_intrinsics.ppx], [0, color_intrinsics.fy, color_intrinsics.ppy], [0, 0, 1]]
    )
    distortion_coefficients = np.asarray(color_intrinsics.coeffs)

    rvec_list, tvec_list = [], []

    img_count = 0
    while True:
        frames = pipeline.wait_for_frames()
        frame = frames.get_color_frame()
        if not frame:
            continue
        frame = np.asarray(frame.get_data())

        use_frame = False
        key = cv2.waitKey(1)
        if key & 0xFF == ord("y"):  # save on pressing 'y'
            use_frame = True
            img_count += 1
        if key & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
        # otherwise process to show the markers
        # Process frames after saving only
        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
        if len(corners) == 0 or matrix_coefficients is None or distortion_coefficients is None:
            continue

        cv2.aruco.drawDetectedMarkers(frame, corners)
        ids = ids.flatten()
        for markerCorner, markerID in zip(corners, ids):
            if markerID != REFERENCE_ID:
                continue
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                markerCorner, MARKER_SIZE, matrix_coefficients, distortion_coefficients
            )
            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, MARKER_SIZE / 2)
            if use_frame:
                rvec_list.append(rvec)
                tvec_list.append(tvec)

        cv2.imshow("img1", frame)  # display the captured image

    print(rvec_list, tvec_list)
    rvecs, tvecs = np.array(rvec_list), np.array(tvec_list)
    rvec, tvec = np.mean(rvecs, axis=0), np.mean(tvecs, axis=0)

    pipeline.stop()
    del pipeline

    return rvec, tvec


def compute_zero(sr, aruco_type, reference_rvec, reference_tvec):
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
    arucoParams = cv2.aruco.DetectorParameters_create()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(sr)
    config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, 30)
    pipeline.start(config)
    frames = pipeline.wait_for_frames()
    frame = frames.get_color_frame()
    color_intrinsics = frame.profile.as_video_stream_profile().intrinsics
    matrix_coefficients = np.array(
        [[color_intrinsics.fx, 0, color_intrinsics.ppx], [0, color_intrinsics.fy, color_intrinsics.ppy], [0, 0, 1]]
    )
    distortion_coefficients = np.asarray(color_intrinsics.coeffs)

    centers = []

    img_count = 0
    while True:
        frames = pipeline.wait_for_frames()
        frame = frames.get_color_frame()
        if not frame:
            continue
        frame = np.asarray(frame.get_data())

        use_frame = False
        key = cv2.waitKey(1)
        if key & 0xFF == ord("y"):  # save on pressing 'y'
            use_frame = True
            img_count += 1
        if key & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
        # otherwise process to show the markers
        # Process frames after saving only
        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
        if len(corners) == 0 or matrix_coefficients is None or distortion_coefficients is None:
            continue

        cv2.aruco.drawDetectedMarkers(frame, corners)
        ids = ids.flatten()
        for markerCorner, markerID in zip(corners, ids):
            if markerID not in CUBE_IDS:
                continue
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                markerCorner, CUBE_MARKER_SIZE, matrix_coefficients, distortion_coefficients
            )
            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, CUBE_MARKER_SIZE / 2)
            if use_frame:
                composedRvec, composedTvec = relativePosition(rvec, tvec, reference_rvec, reference_tvec)
                # Now compute the actual center of the marker cube
                cube_center = np.array([[0, 0, -CUBE_MARKER_SIZE / 2]]).T
                rot_mat, _ = cv2.Rodrigues(composedRvec)
                cube_center = rot_mat.dot(cube_center) + composedTvec
                centers.append(cube_center)
                print("printing")
                print(cube_center)
        cv2.imshow("img1", frame)  # display the captured image

    predicted_center = np.mean(np.array(centers), axis=0).squeeze(-1)
    # center + offset = desired zero
    offset = DESIRED_ZERO - predicted_center
    print("for sr", sr, "use offset", offset)

    pipeline.stop()
    del pipeline

    return


def save_coefficients(mtx, dist, rvec, tvec, path):
    """Save the camera matrix and the distortion coefficients to given path/file."""
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    cv_file.write("R", rvec)
    cv_file.write("T", tvec)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()


def load_coefficients(path):
    """Loads camera matrix and distortion coefficients."""
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()
    rvec = cv_file.getNode("R").mat()
    tvec = cv_file.getNode("T").mat()

    cv_file.release()
    return [camera_matrix, dist_matrix, rvec, tvec]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera calibration")
    parser.add_argument(
        "--image-dir", type=str, required=False, default="../output/calibration", help="image directory path"
    )
    parser.add_argument(
        "--square-size", type=float, required=False, default=0.0217, help="chessboard square size. Specifically tuned."
    )
    parser.add_argument("--width", type=int, required=False, default=9, help="chessboard width size, default is 9")
    parser.add_argument("--height", type=int, required=False, default=6, help="chessboard height size, default is 6")
    parser.add_argument(
        "--save-file",
        type=str,
        required=False,
        default="calibration.yaml",
        help="YML file to save calibration matrices",
    )

    args = parser.parse_args()

    rvec_1, tvec_1 = calibrate_camera(SRs[0], "DICT_4X4_50")
    compute_zero(SRs[0], "DICT_4X4_50", rvec_1, tvec_1)
    rvec_2, tvec_2 = calibrate_camera(SRs[1], "DICT_4X4_50")
    compute_zero(SRs[1], "DICT_4X4_50", rvec_2, tvec_2)
    save_coefficients(rvec_1, tvec_1, rvec_2, tvec_2, args.save_file)
