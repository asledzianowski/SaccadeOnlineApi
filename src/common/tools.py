import time
import cv2 as cv
import numpy as np
from math import atan2, degrees


def check_fps(video):

    print('FPS count check')
    # Start default camera
    #video = cv.VideoCapture(0);

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv.__version__).split('.')

    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.

    if int(major_ver) < 3:
        fps = video.get(cv.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = video.get(cv.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    # Number of frames to capture
    num_frames = 120;

    print("Capturing {0} frames".format(num_frames), ", please wait...")

    # Start time
    start = time.time()

    # Grab a few frames
    for i in range(0, num_frames):
        ret, frame = video.read()

    # End time
    end = time.time()

    # Time elapsed
    seconds = end - start
    print("Time taken : {0} seconds".format(seconds))

    # Calculate frames per second
    fps = num_frames / seconds
    print("Estimated frames per second : {0}".format(fps))

    # Release video
    video.release()


def degrees_to_pixels(monitor_width_cm, monitor_horizontal_resolution_px, distance_from_monitor_cm, size_in_deg):
    deg_per_px = degrees(atan2(.5 * monitor_width_cm, distance_from_monitor_cm)) / (.5 * monitor_horizontal_resolution_px)
    size_in_px = size_in_deg / deg_per_px
    return size_in_px


def calculate_screen_data(screen_width_px, screen_height_px, screen_width_mm,
                          dist_from_screen_cm, target_to_fix_distance_deg):

    screen_width_cm = screen_width_mm / 10
    target_dist = degrees_to_pixels(screen_width_cm, screen_width_px, dist_from_screen_cm,
                                    target_to_fix_distance_deg)

    screen_data = {}
    screen_data['screen_width'] = screen_width_px
    screen_data['screen_height'] = screen_height_px
    screen_data['horizontal_center'] = horizontal_center = int(np.round(screen_width_px / 2))
    screen_data['vertical_center'] = int(np.round(screen_height_px / 2))
    screen_data['target_dist'] = target_dist
    screen_data['target_horiz_left_pos'] = int(np.round(horizontal_center - target_dist))
    screen_data['target_horiz_right_pos'] = int(np.round(horizontal_center + target_dist))
    return screen_data