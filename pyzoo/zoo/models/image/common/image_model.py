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

from zoo.models.common import ZooModel
from zoo.models.image.common.image_config import ImageConfigure
from zoo.feature.image.imageset import *

if sys.version >= '3':
    long = int
    unicode = str


class ImageModel(ZooModel):
    """
    The basic class for image model.
    """
    def predict_image_set(self, image, configure=None):
        res = callBigDlFunc(self.bigdl_type, "imageModelPredict", self.value,
                            image, configure)
        return ImageSet(res)

    def get_config(self):
        config = callBigDlFunc(self.bigdl_type, "getImageConfig", self.value)
        return ImageConfigure(jvalue=config)
