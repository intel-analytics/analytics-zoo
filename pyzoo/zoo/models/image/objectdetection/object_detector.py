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
from bigdl.transform.vision.image import FeatureTransformer

from zoo.models.image.common.image_model import ImageModel
from zoo.feature.image.imageset import *
from zoo.feature.image.imagePreprocessing import *

if sys.version >= '3':
    long = int
    unicode = str


def read_pascal_label_map():
    """
    load pascal label map
    """
    return callBigDlFunc("float", "readPascalLabelMap")


def read_coco_label_map():
    """
    load coco label map
    """
    return callBigDlFunc("float", "readCocoLabelMap")


class ObjectDetector(ImageModel):
    """
    A pre-trained object detector model.

    :param model_path The path containing the pre-trained model
    """
    def __init__(self, bigdl_type="float"):
        self.bigdl_type = bigdl_type
        super(ObjectDetector, self).__init__(None, bigdl_type)

    @staticmethod
    def load_model(path, weight_path=None, bigdl_type="float"):
        """
        Load an existing object detection model (with weights).

        # Arguments
        path: The path to save the model. Local file system, HDFS and Amazon S3 are supported.
              HDFS path should be like 'hdfs://[host]:[port]/xxx'.
              Amazon S3 path should be like 's3a://bucket/xxx'.
        """
        jmodel = callBigDlFunc(bigdl_type, "loadObjectDetector", path, weight_path)
        model = ImageModel._do_load(jmodel, bigdl_type)
        model.__class__ = ObjectDetector
        return model


class ImInfo(ImagePreprocessing):
    """
    Generate imInfo
    imInfo is a tensor that contains height, width, scaleInHeight, scaleInWidth
    """
    def __init__(self, bigdl_type="float"):
        super(ImInfo, self).__init__(bigdl_type)

class DummyGT(ImagePreprocessing):
    """
    
    """
    def __init__(self, bigdl_type="float"):
        super(DummyGT, self).__init__(bigdl_type)


class DecodeOutput(ImagePreprocessing):
    """
    Decode the detection output
    The output of the model prediction is a 1-dim tensor
    The first element of tensor is the number(K) of objects detected,
    followed by [label score x1 y1 x2 y2] X K
    For example, if there are 2 detected objects, then K = 2, the tensor may
    looks like
    ```2, 1, 0.5, 10, 20, 50, 80, 3, 0.3, 20, 10, 40, 70```
    After decoding, it returns a 2-dim tensor, each row represents a detected object
    ```
    1, 0.5, 10, 20, 50, 80
    3, 0.3, 20, 10, 40, 70
    ```
    """
    def __init__(self, bigdl_type="float"):
        super(DecodeOutput, self).__init__(bigdl_type)


class ScaleDetection(ImagePreprocessing):
    """
    If the detection is normalized, for example, ssd detected bounding box is in [0, 1],
    need to scale the bbox according to the original image size.
    Note that in this transformer, the tensor from model output will be decoded,
    just like `DecodeOutput`
    """
    def __init__(self, bigdl_type="float"):
        super(ScaleDetection, self).__init__(bigdl_type)


class Visualizer(ImagePreprocessing):
    """
    Visualizer is a transformer to visualize the detection results
    (tensors that encodes label, score, boundingbox)
    You can call image_frame.get_image() to get the visualized results
    """
    def __init__(self, label_map, thresh=0.3, encoding="png",
                 bigdl_type="float"):
        self.value = callBigDlFunc(
            bigdl_type, JavaValue.jvm_class_constructor(self), label_map, thresh, encoding)

    def __call__(self, image_set, bigdl_type="float"):
        """
        transform ImageSet
        """
        jset = callBigDlFunc(bigdl_type,
                             "transformImageSet", self.value, image_set)
        return ImageSet(jvalue=jset)
