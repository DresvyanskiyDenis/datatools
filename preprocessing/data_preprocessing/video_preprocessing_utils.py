#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Contains functions for video preprocessing.

List of functions:

    * extract_frames_from_videofile - extracts image frames from provided videofile.
    * extract_frames_from_all_videos_in_dir - extracts image frames from all videofiles located in probided directory.
"""

import os
import re
from typing import Optional

import cv2


__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"


def extract_frames_from_videofile(input_path:str, output_path:str, every_n_frame:Optional[int]=None)->None:
    """Extracts frames from videofile. To extract not all frames, but every n, specify the parameter every_n_frame.

    :param input_path: str
            path to the videofile.
    :param output_path: str
            path to the directory, where extracted frames will be saved.
    :param every_n_frame: int
            specify this parameter, if you need to extract only every n frame.
    :return: None
    """
    videofile=cv2.VideoCapture(input_path)
    # check if videofile was opened:
    if videofile is None or not videofile.isOpened():
        raise Exception('Warning: unable to open video source: ', input_path)
    #calculate the name of videofile and its extention
    video_filename=re.split(r'\\|/',input_path)[-1].split('.')[0]
    file_extention=re.split(r'\\|/',input_path)[-1].split('.')[-1]
    video_frame_rate=videofile.get(cv2.CAP_PROP_FPS)
    # creating directories for saving files
    if not os.path.exists(os.path.join(output_path,video_filename,video_filename+'_frames')):
        os.makedirs(os.path.join(output_path,video_filename,video_filename+'_frames'), exist_ok=True)
    currentframe=-1 # we will start from frame with number 0 (see next, in few lines, we increment currentframe
                    # right after reading it)
    # if every_n_frame is defined, skip every n frames
    every_frame=every_n_frame if not every_n_frame is None else 1
    while (True):
        # reading from frame
        ret, frame = videofile.read()
        currentframe += 1
        # if currentframe is not integer divisible by every_frame, skip it
        if not currentframe%every_frame==0:
            continue
        if ret:
            # if video is still left continue creating images
            current_filename = video_filename + str(currentframe) + '.jpg'
            # writing the extracted images
            cv2.imwrite(os.path.join(output_path,video_filename, video_filename+'_frames', current_filename), frame)
        else:
            break
    # write meta data of videofile
    with open(os.path.join(output_path,video_filename, "%s_%s.txt"%(video_filename, 'metadata')), "w") as metadata:
        metadata.write('filename:%s\n'
                       'frame_rate:%f\n'%(video_filename+"."+file_extention, video_frame_rate))
    # Release all space and windows once done
    videofile.release()
    cv2.destroyAllWindows()


def extract_frames_from_all_videos_in_dir(input_dir:str, output_dir:str, every_n_frame:Optional[int]=None)-> None:
    """Extracts frames from all videofiles located in input_dir.
    To extract not all frames, but every n, specify the parameter every_n_frame.

    :param input_dir: str
            path to the directory with videofiles.
    :param output_dir: str
            path to the directory, where extracted frames will be saved.
    :param every_n_frame: int
            specify this parameter, if you need to extract only every n frame.
    :return: None
    """
    absolute_paths_to_videos=[]
    # walk through all the subdirectories and find all the videofiles and their absolute paths
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith((".avi",".mp4")):
                absolute_paths_to_videos.append(os.path.join(root, file))
    # create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # extract frames file-by-file
    for absolute_path_to_video in absolute_paths_to_videos:
        extract_frames_from_videofile(input_path=absolute_path_to_video,
                                      output_path=output_dir, every_n_frame=every_n_frame)

