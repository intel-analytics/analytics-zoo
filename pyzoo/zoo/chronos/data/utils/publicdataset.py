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
import time
import requests

import pandas as pd
from zoo.chronos.data.tsdataset import TSDataset

NETWORK_TRAFFIC_DATA = ['2018'+str(i).zfill(2) for i in range(1, 13)] + [
    '2019'+str(i).zfill(2) for i in range(1, 13)]

BASE_URL = {'network_traffic': [
    f'http://mawi.wide.ad.jp/~agurim/dataset/{val}/{val}.agr' for val in NETWORK_TRAFFIC_DATA]}


class PublicDataset:

    def __init__(self, **kwargs):
        self.name = kwargs['name']
        self.redownload = kwargs['redownload']

        self.__abspath = os.path.join(
            os.path.expanduser(kwargs['path']), self.name)
        self.__data_path = os.path.join(
            self.__abspath, self.name + '_data.csv')

    def get_public_data(self, chunk_size=1024, progress_bar=True):
        """
        param chunk_size: Byte size of a single download, preferably an integer multiple of 2.
        param progress_bar: Set the progress bar to display when downloading, default True.
        """
        assert isinstance(
            chunk_size, int), "chunk_size must be a int type."
        if self.redownload:
            exists_file = os.listdir(self.__abspath)
            _ = [os.remove(os.path.join(self.__abspath, x))
                 for x in exists_file if x in NETWORK_TRAFFIC_DATA]
        if not os.path.exists(self.__abspath):
            os.makedirs(self.__abspath)
        url = BASE_URL[self.name]
        if isinstance(BASE_URL[self.name], list):
            for val in url:
                download(val, self.__abspath, chunk_size)
        else:
            download(url, self.__abspath, chunk_size)
        return self

    def preprocess_network_traffic(self):
        """ 
        preprocess_network_traffic will match the Starttime and endtime(avgrate, total)
        of data accordingto the regularity, and generate a csv file, the file name 
        is network_traffic_data.csv
        return partially preprocessed tsdata.
        """
        _is_first_columns = True
        pattern = r"%Sta.*?\((.*?)\)\n%%End.*?\((.*?)\)\n%Avg.*?\s(\d+\.\w+).*?\n%total:\s(\d+)"

        for val in NETWORK_TRAFFIC_DATA:
            with open(os.path.join(self.__abspath, val), 'r') as f:
                content = f.read()
                result = re.findall(pattern, content, re.DOTALL)
            columns_name = ['StartTime', 'EndTime', 'AvgRate', 'total']
            raw_df = pd.DataFrame(columns=columns_name, data=result)
            raw_df.to_csv(self.__data_path, mode='a',
                          header=_is_first_columns, index=False, chunksize=256)
            _is_first_columns = False

        self.df = pd.DataFrame(pd.to_datetime(raw_df.StartTime))
        raw_df.AvgRate.str[-4:].unique()
        self.df['AvgRate'] = raw_df.AvgRate.apply(lambda x: float(
            x[:-4]) if x.endswith("Mbps") else float(x[:-4])*1000)
        self.df["total"] = raw_df["total"]
        return TSDataset.from_pandas(self.df, dt_col="StartTime", target_col=["AvgRate", "total"],
                                     with_split=True, test_ratio=0.1)

    def preprocess_zip_file(self):
        pass


def download(url, path, chunk_size):
    """
    param url: File download source address,can be a str or a list.
    param path: File save path.
    """
    start_time = time.time()
    req = requests.get(url, stream=True)
    size, content_size = 0, int(req.headers['content-length'])
    try:
        if req.status_code == 200:
            pass
            # print('Start download,[file_size]:{size:.2f}MB'.format(size=content_size/chunk_size/1024))
    except Exception:
        raise RuntimeError('download failure.')
    file_name = url.split('/')[-1].partition('.')[0]
    with open(os.path.join(path, file_name), 'wb') as f:
        for chunk in req.iter_content(1024 * chunk_size):
            if chunk:
                f.write(chunk)
                size += len(chunk)
                print('\r'+'file %s:%s%.2f%%' % (file_name, '>'*int(size *
                      50/content_size), float(size/content_size*100)), end='')
                f.flush()
        print('')
