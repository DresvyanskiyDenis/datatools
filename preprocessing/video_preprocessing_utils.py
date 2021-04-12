#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: write description of module
"""

import os
import cv2
import re

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"


def extract_frames_from_videofile(input_path:str, output_path:str)->None:
    # TODO: add description
    videofile=cv2.VideoCapture(input_path)
    # check if videofile was opened:
    if videofile is None or not videofile.isOpened():
        raise Exception('Warning: unable to open video source: ', input_path)
    #calculate the name of videofile
    video_filename=re.split(r'\\|/',input_path)[-1].split('.')[0]
    video_frame_rate=videofile.get(cv2.CAP_PROP_FPS)
    # creating directories for saving files
    if not os.path.exists(os.path.join(output_path,video_filename,video_filename+'_frames')):
        os.makedirs(os.path.join(output_path,video_filename,video_filename+'_frames'), exist_ok=True)
    currentframe=0
    while (True):
        # reading from frame
        ret, frame = videofile.read()

        if ret:
            # if video is still left continue creating images
            current_filename = video_filename + str(currentframe) + '.jpg'
            # writing the extracted images
            cv2.imwrite(os.path.join(output_path,video_filename, video_filename+'_frames', current_filename), frame)
            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break
    # write meta data of videofile
    with open(os.path.join(output_path,video_filename, "%s_%s.txt"%(video_filename, 'metadata'))
            , "w") as metadata:
        metadata.write('filename:%s\n'
                       'frame_rate:%f\n'%(video_filename, video_frame_rate))
    # Release all space and windows once done
    videofile.release()
    cv2.destroyAllWindows()


def extract_frames_from_all_videos_in_dir(input_dir:str, output_dir:str)-> None:
    # TODO: add description
    list_video_filenames=os.listdir(input_dir)
    # create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # extract frames file-by-file
    for video_filename in list_video_filenames:
        extract_frames_from_videofile(input_path=os.path.join(input_dir,video_filename),
                                      output_path=os.path.join(output_dir,video_filename))



if __name__ == '__main__':
    input_path=r'D:\Databases\DAiSEE\DAiSEE\DataSet\Train\181374\181374015\181374015.avi'
    output_path=r'C:\tmp'
    extract_frames_from_videofile(input_path, output_path)