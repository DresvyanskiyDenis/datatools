#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module contains functions for working with archives.

List of functions:

    * _extract_files_by_provided_names_tar - extracts the members of archive provided in names.
    * extract_files_from_tar_file_with_patterns - extracts the part of the archive in accordance to the provided
      pattern (or patterns)
"""
__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

import tarfile
from typing import List
import os


def _extract_files_by_provided_names_tar(t_file:tarfile.TarFile, names:List[str], output_path:str)->None:
    """Extracts the members of archive by provided names.

    :param t_file:
    :param names: List[str]
            List of members (parts) of the archive to be extracted.
    :param output_path: str
            Path to the extraction.
    :return: None
    """
    for name in names:
        member=t_file.getmember(name)
        extracting_path=output_path
        t_file.extract(member, path=extracting_path)

def extract_files_from_tar_file_with_patterns(path_to_tar_file:str, output_path:str, patterns:List[str])->None:
    """Extracts files from the archive by provided patterns.

    :param path_to_tar_file: str
            Path to the archive.
    :param output_path: str
            Path to the output directory
    :param patterns: List[str]
            List of str patterns to be extracted. Files that match this patter will be extracted.
            (for example .csv, 1.csv and so on)
    :return: None
    """
    # check if output path exists
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # get information about members inside archive
    with tarfile.open(path_to_tar_file, 'r') as t_file:
        names=t_file.getnames()
        #info[1000].name
        # iterate through patterns and extract all compatible files
        for pattern in patterns:
            # find compatible to the pattern files
            compatible_names=list(filter(lambda x:pattern in x, names))
            # extract them
            _extract_files_by_provided_names_tar(t_file, compatible_names, output_path)
