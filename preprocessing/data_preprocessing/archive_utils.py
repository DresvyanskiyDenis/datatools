'''
# TODO: write description
'''

import tarfile
from typing import List
import os
import numpy as np


def _extract_files_by_provided_names_tar(t_file:tarfile.TarFile, names:List[str], output_path:str)->None:
    # TODO: write description
    for name in names:
        member=t_file.getmember(name)
        extracting_path=output_path
        t_file.extract(member, path=extracting_path)

def extract_files_from_tar_file_with_patterns(path_to_tar_file:str, output_path:str, patterns:List[str])->None:
    # TODO: write description
    # check if output path exists
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # get information about members inside archive
    with tarfile.open(path_to_tar_file, 'r') as t_file:
        names=t_file.getnames()
        #info[1000].name
        # iterate through patterns and extract all compatible files
        for pattern in patterns:
            # find compatible to the patter files
            compatible_names=list(filter(lambda x:pattern in x, names))
            # extract them
            _extract_files_by_provided_names_tar(t_file, compatible_names, output_path)





if __name__=='__main__':
    path_to_file=r'D:\NoXi.tar'
    output_path=r'D:\Noxi_extracted'
    patterns=['Expert_video.mp4','Novice_video.mp4','Novice_close.wav', 'Expert_close.wav', 'metadata.ini', 'Expert_session_metadata.ini', 'Novice_session_metadata.ini']
    extract_files_from_tar_file_with_patterns(path_to_file, output_path, patterns)