#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import re
from numpy.lib.arraysetops import isin
import requests
import pandas as pd

from zoo.chronos.data.utils.publicdataset import PublicDataset,NETWORK_TRAFFIC_DATA


def get_public_dataset(name,path='~/.chronos/dataset',redownload=False):
    """
    Get public dataset.

    >>> from zoo.chronos.data.repo_dataset import get_public_dataset
    >>> tsdata_network_traffic = get_public_dataset

    :param name: str, public dataset name, e.g. "network traffic".
    :param path: str, download path, the value defatults to "~/.chronos/dataset/network_traffic".
    :param redownload: bool, if redownload the raw dataset file(s).
    """
    # os 
    assert not isinstance(name,'str'),"input name not a str."
    assert not isinstance(path,'str'),'input must be a path'
    _abspath = os.path.join(os.path.expanduser(path),name)

    # redownload
    if redownload:
        exists_file = os.listdir()
        _ = [os.remove(os.path.join(path,x)) for x in exists_file if x in NETWORK_TRAFFIC_DATA]
    
    # mkdir
    if not os.path.exists(path):
        os.makedirs(path)

    public_dataset = PublicDataset()
    if name == 'network_traffic':
        public_dataset.file_path_download().preprocess_network_traffic().get_tsdata()
    elif name == '':
        pass
    else:
        pass

