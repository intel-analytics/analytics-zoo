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
from zoo.models.image.imageclassification import ImageClassifier
from zoo.pipeline.api.keras.layers import Activation
from zoo.pipeline.api.keras.models import Sequential
from zoo.feature.image import ImageSet
from zoo.common.nncontext import init_nncontext
from optparse import OptionParser


def predict(model_path, image_path, top_n):
    sc = init_nncontext("Image classification inference example using int8 quantized model")
    images = ImageSet.read(image_path, sc, image_codec=1)
    part_model = ImageClassifier.load_model(model_path)
    output = part_model.predict_image_set(images)
    label_map = part_model.get_config().label_map()

    # list of images composing uri and results in tuple format
    predicts = output.get_predict().collect()

    model = Sequential().add(part_model)\
        .add(Activation("softmax", input_shape=predicts[0][1][0].shape))
    for result in predicts:
        (uri, probs) = result
        out = model.forward(probs[0])
        sorted_probs = [(prob, index) for index, prob in enumerate(out)]
        sorted_probs.sort()
        print("Image : %s, top %d prediction result" % (uri, top_n))
        for i in range(top_n):
            print("\t%s, %f" % (label_map[sorted_probs[999 - i][1]], sorted_probs[999 - i][0]))


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--model", type=str, dest="model_path",
                      help="The path to the downloaded int8 model snapshot")
    parser.add_option("--image", type=str, dest="image_path",
                      help="The local folder path that contains images for prediction")
    parser.add_option("--topN", type=int, dest="topN", default=5,
                      help="The top N classes with highest probabilities as output")

    (options, args) = parser.parse_args(sys.argv)
    predict(options.model_path, options.image_path, options.topN)
