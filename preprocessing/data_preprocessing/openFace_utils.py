#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: write description of module
"""

import shutil
import pandas as pd
import numpy as np
import subprocess
import os

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"



def extract_openface_FAU_from_images_in_dir(path_to_dir:str, path_to_extractor:str)->pd.DataFrame:
    # TODO: write description
    tmp_dir='tmp_dir'
    # check if output_dir exists
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
    command_line=path_to_extractor+' -fdir '+path_to_dir+' -out_dir '+tmp_dir+' -aus'
    subprocess.run(command_line)
    # read extracted features and save them in dataframe
    result_df=[]
    filenames=os.listdir(tmp_dir)
    filenames=[filename for filename in filenames if filename.split('.')[-1]=='csv']
    for filename in filenames:
        df=pd.read_csv(os.path.join(tmp_dir, filename))
        result_df.append(df)
    # concatenate obtained dataframes
    result_df=pd.concat(result_df, axis=0)
    result_df['filename']=filenames
    result_df.columns=[column.strip() for column in result_df.columns]
    result_df=result_df.drop(columns=['face', 'confidence'])
    # delete tmp dir with all files
    shutil.rmtree(tmp_dir)
    return result_df




if __name__=="__main__":
    path_to_dir=r'E:\Databases\DAiSEE\DAiSEE\dev_processed\extracted_faces\556463012'
    path_to_output=r'E:\Databases\DAiSEE\extracted_openface_features'
    path_to_extractor='../../../OpenFace\\FaceLandmarkImg.exe'
    features=extract_openface_FAU_from_images_in_dir(path_to_dir, path_to_extractor)
    a=1+2