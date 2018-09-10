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

import sys
from bigdl.util.common import JavaValue
from bigdl.util.common import callBigDlFunc

from zoo.feature.common import Preprocessing

if sys.version >= '3':
    long = int
    unicode = str


class ImageConfigure(JavaValue):
    """
    predictor configure
    :param pre_processor preprocessor of ImageSet before model inference
    :param post_processor postprocessor of ImageSet after model inference
    :param batch_per_partition batch size per partition
    :param label_map mapping from prediction result indexes to real dataset labels
    :param feature_padding_param featurePaddingParam if the inputs have variant size
    """
    def __init__(self, pre_processor=None,
                 post_processor=None,
                 batch_per_partition=4,
                 label_map=None, feature_padding_param=None, jvalue=None, bigdl_type="float"):
        self.bigdl_type = bigdl_type
        if jvalue:
            self.value = jvalue
        else:
            if pre_processor:
                assert issubclass(pre_processor.__class__, Preprocessing), \
                    "the pre_processor should be subclass of Preprocessing"
            if post_processor:
                assert issubclass(post_processor.__class__, Preprocessing), \
                    "the post_processor should be subclass of Preprocessing"
            self.value = callBigDlFunc(
                bigdl_type, JavaValue.jvm_class_constructor(self),
                pre_processor,
                post_processor,
                batch_per_partition,
                label_map,
                feature_padding_param)

    def label_map(self):
        return callBigDlFunc(self.bigdl_type, "getLabelMap", self.value)


class PaddingParam(JavaValue):

    def __init__(self, bigdl_type="float"):
        self.value = callBigDlFunc(
            bigdl_type, JavaValue.jvm_class_constructor(self))
