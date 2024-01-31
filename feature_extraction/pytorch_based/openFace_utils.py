#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utils for working with OpenFace library

Module contains functions for using OpenFace lib.
OpenFace:https://github.com/TadasBaltrusaitis/OpenFace

List of functions:

    * extract_openface_FAU_from_images_in_dir - runs OpenFace lib to extract FAU from all images located in provided directory.
"""
__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

import shutil
from typing import Optional
import pandas as pd
import subprocess
import os


def extract_openface_FAU_from_images_in_dir(path_to_dir:str, path_to_extractor:str)->pd.DataFrame:
    """Runs OpenFace toolkit to extract Facial Action Units from all images located in directory.

    :param path_to_dir: str
            path to directory with images
    :param path_to_extractor: str
            path to the toolkit (.exe file), which will be run via python command line
    :return: Optional[pd.DataFrame]
            DataFrame with results of the extraction (FAUs). All paths to the images will be saved.
    """
    tmp_dir='tmp_dir'
    # check if output_dir exists
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
    command_line=path_to_extractor+' -fdir '+path_to_dir+' -out_dir '+tmp_dir+' -aus'
    subprocess.run(command_line)
    # read extracted features and save them in dataframe
    result_df=[]
    filenames=os.listdir(tmp_dir)
    filenames = [filename for filename in filenames if filename.split('.')[-1] == 'csv']
    if len(filenames)==0:
        return None
    for filename in filenames:
        df=pd.read_csv(os.path.join(tmp_dir, filename))
        df['filename']=filename
        max_confident_face_features_idx=df[' confidence'].idxmax()
        result_df.append(df.iloc[max_confident_face_features_idx, :])
    # concatenate obtained dataframes
    result_df=pd.concat(result_df, axis=1).T
    result_df['filename']=result_df['filename'].apply(lambda x: x.replace('.csv',''))
    result_df.columns=[column.strip() for column in result_df.columns]
    result_df=result_df.drop(columns=['face', 'confidence'])
    # delete tmp dir with all files
    shutil.rmtree(tmp_dir)
    return result_df




if __name__=="__main__":
    path_to_dir=r'E:\Databases\DAiSEE\DAiSEE\dev_processed\extracted_faces\556463012'
    path_to_output=r'E:\Databases\DAiSEE\extracted_openface_features'
    path_to_extractor= '../../../OpenFace/FaceLandmarkImg.exe'
    features=extract_openface_FAU_from_images_in_dir(path_to_dir, path_to_extractor)