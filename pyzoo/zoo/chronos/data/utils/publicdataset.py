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
import requests
import pandas as pd
from zoo.chronos.data.tsdataset import TSDataset

NETWORK_TRAFFIC_DATA = ['2018'+str(i).zfill(2) for i in range(1, 13)] + [
    '2019'+str(i).zfill(2) for i in range(1, 13)]
BASE_URL = {'network_traffic': [
    f'http://mawi.wide.ad.jp/~agurim/dataset/{val}/{val}.agr' for val in NETWORK_TRAFFIC_DATA]}

class PublicDataset:

    def __init__(self,name,path,redownload):
        self.name = name
        self.path = path
        self.redownload=redownload

        # data
        self.data_path = os.path.join(self.path,self.name + '_data.csv')

    
    def file_path_download(self, chunk_size=1024):
        """

        """
        url = BASE_URL[self.name]
        req = requests.get(url, stream=True)
        file_name = url.split('/')[-1].partition('.')[0]
        with open(os.path.join(self.path, file_name), 'wb') as f:
            for chunk in req.iter_content(1024 * chunk_size):
                if chunk:
                    f.write(chunk)
                    f.flush()
        return self


    # network_traffic    
    def preprocess_network_traffic(self):
        """

        """
        _is_first_columns = True
        pattern = r"%Sta.*?\((.*?)\)\n%%End.*?\((.*?)\)\n%Avg.*?\s(\d+\.\w+).*?\n%total:\s(\d+)"
        data_path = os.path.join(self.path, 'data.csv')
        if not os.path.exists(data_path):
            for val in NETWORK_TRAFFIC_DATA:
                with open(os.path.join(self.path, val), 'r') as f:
                    content = f.read()
                    result = re.findall(pattern, content, re.DOTALL)
                columns_name = ['StartTime', 'EndTime', 'AvgRate', 'total']
                self.raw_df = pd.DataFrame(columns=columns_name, data=result)
                self.raw_df.to_csv(self.data_path, mode='a',
                            header=_is_first_columns, index=False, chunksize=256)
                _is_first_columns = False

        return self

    def get_tsdata(self):
        """
        
        """
        raw_df = pd.read_csv(self.data_path)
        df = pd.DataFrame(pd.to_datetime(raw_df.StartTime))
        raw_df.AvgRate.str[-4:].unique()
        df['AvgRate'] = raw_df.AvgRate.apply(lambda x: float(
            x[:-4]) if x.endswith("Mbps") else float(x[:-4])*1000)
        df["total"] = raw_df["total"]
        return TSDataset.from_pandas(df, dt_col="StartTime", target_col=["AvgRate", "total"],
                                    with_split=True, test_ratio=0.1)
