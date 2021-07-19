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
from optparse import OptionParser
from math import ceil
from datetime import datetime

from zoo.common.nncontext import *
from zoo.feature.image import *
from zoo.pipeline.nnframes import *
from zoo.pipeline.estimator import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.nn.initialization_method import *


def config_option_parser():
    parser = OptionParser()
    parser.add_option("-f", "--folder", type=str, dest="folder", default="",
                      help="url of hdf+s folder store the hadoop sequence files")
    parser.add_option("--model", type=str, dest="model", default="", help="model snapshot location")
    parser.add_option("--state", type=str, dest="state", default="", help="state snapshot location")
    parser.add_option("--checkpoint", type=str, dest="checkpoint", default="",
                      help="where to cache the model")
    parser.add_option("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                      help="overwrite checkpoint files")
    parser.add_option("-e", "--maxEpoch", type=int, dest="maxEpoch", default=0,
                      help="epoch numbers")
    parser.add_option("-i", "--maxIteration", type=int, dest="maxIteration", default=3100,
                      help="iteration numbers")
    parser.add_option("-l", "--learningRate", type=float, dest="learningRate", default=0.01,
                      help="learning rate")
    parser.add_option("--warmupEpoch", type=int, dest="warmupEpoch", default=0,
                      help="warm up epoch numbers")
    parser.add_option("--maxLr", type=float, dest="maxLr", default=0.0, help="max Lr after warm up")
    parser.add_option("-b", "--batchSize", type=int, dest="batchSize", help="batch size")
    parser.add_option("--classNum", type=int, dest="classNum", default=1000, help="class number")
    parser.add_option("--weightDecay", type=float, dest="weightDecay", default=0.0001,
                      help="weight decay")
    parser.add_option("--checkpointIteration", type=int, dest="checkpointIteration", default=620,
                      help="checkpoint interval of iterations")
    parser.add_option("--gradientMin", type=float, dest="gradientMin", default=0.0,
                      help="min gradient clipping by")
    parser.add_option("--gradientMax", type=float, dest="gradientMax", default=0.0,
                      help="max gradient clipping by")
    parser.add_option("--gradientL2NormThreshold", type=float, dest="gradientL2NormThreshold",
                      default=0.0, help="gradient L2-Norm threshold")
    parser.add_option("--memoryType", type=str, dest="memoryType", default="DRAM",
                      help="memory storage type, DRAM or PMEM")

    return parser


def get_inception_data(url, sc=None, data_type="train"):
    path = os.path.join(url, data_type)
    return SeqFileFolder.files_to_image_frame(url=path, sc=sc, class_num=1000)


def t(input_t):
    if type(input_t) is list:
        # insert into index 0 spot, such that the real data starts from index 1
        temp = [0]
        temp.extend(input_t)
        return dict(enumerate(temp))
    # if dictionary, return it back
    return input_t


def inception_layer_v1(input_size, config, name_prefix=""):
    concat = Concat(2)
    conv1 = Sequential()
    conv1.add(SpatialConvolution(input_size, config[1][1], 1, 1, 1, 1)
              .set_init_method(weight_init_method=Xavier(), bias_init_method=ConstInitMethod(0.1))
              .set_name(name_prefix + "1x1"))
    conv1.add(ReLU(True).set_name(name_prefix + "relu_1x1"))
    concat.add(conv1)
    conv3 = Sequential()
    conv3.add(SpatialConvolution(input_size, config[2][1], 1, 1, 1, 1)
              .set_init_method(weight_init_method=Xavier(), bias_init_method=ConstInitMethod(0.1))
              .set_name(name_prefix + "3x3_reduce"))
    conv3.add(ReLU(True).set_name(name_prefix + "relu_3x3_reduce"))
    conv3.add(SpatialConvolution(config[2][1], config[2][2], 3, 3, 1, 1, 1, 1)
              .set_init_method(weight_init_method=Xavier(), bias_init_method=ConstInitMethod(0.1))
              .set_name(name_prefix + "3x3"))
    conv3.add(ReLU(True).set_name(name_prefix + "relu_3x3"))
    concat.add(conv3)
    conv5 = Sequential()
    conv5.add(SpatialConvolution(input_size, config[3][1], 1, 1, 1, 1)
              .set_init_method(weight_init_method=Xavier(), bias_init_method=ConstInitMethod(0.1))
              .set_name(name_prefix + "5x5_reduce"))
    conv5.add(ReLU(True).set_name(name_prefix + "relu_5x5_reduce"))
    conv5.add(SpatialConvolution(config[3][1], config[3][2], 5, 5, 1, 1, 2, 2)
              .set_init_method(weight_init_method=Xavier(), bias_init_method=ConstInitMethod(0.1))
              .set_name(name_prefix + "5x5"))
    conv5.add(ReLU(True).set_name(name_prefix + "relu_5x5"))
    concat.add(conv5)
    pool = Sequential()
    pool.add(SpatialMaxPooling(3, 3, 1, 1, 1, 1, to_ceil=True).set_name(name_prefix + "pool"))
    pool.add(SpatialConvolution(input_size, config[4][1], 1, 1, 1, 1)
             .set_init_method(weight_init_method=Xavier(), bias_init_method=ConstInitMethod(0.1))
             .set_name(name_prefix + "pool_proj"))
    pool.add(ReLU(True).set_name(name_prefix + "relu_pool_proj"))
    concat.add(pool).set_name(name_prefix + "output")
    return concat


def inception_v1_no_aux_classifier(class_num, has_dropout=True):
    model = Sequential()
    model.add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, False)
              .set_init_method(weight_init_method=Xavier(), bias_init_method=ConstInitMethod(0.1))
              .set_name("conv1/7x7_s2"))
    model.add(ReLU(True).set_name("conv1/relu_7x7"))
    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool1/3x3_s2"))
    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75).set_name("pool1/norm1"))
    model.add(SpatialConvolution(64, 64, 1, 1, 1, 1)
              .set_init_method(weight_init_method=Xavier(), bias_init_method=ConstInitMethod(0.1))
              .set_name("conv2/3x3_reduce"))
    model.add(ReLU(True).set_name("conv2/relu_3x3_reduce"))
    model.add(SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1)
              .set_init_method(weight_init_method=Xavier(), bias_init_method=ConstInitMethod(0.1))
              .set_name("conv2/3x3"))
    model.add(ReLU(True).set_name("conv2/relu_3x3"))
    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75).set_name("conv2/norm2"))
    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True).set_name("pool2/3x3_s2"))
    model.add(inception_layer_v1(192, t([t([64]), t(
        [96, 128]), t([16, 32]), t([32])]), "inception_3a/"))
    model.add(inception_layer_v1(256, t([t([128]), t(
        [128, 192]), t([32, 96]), t([64])]), "inception_3b/"))
    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True))
    model.add(inception_layer_v1(480, t([t([192]), t(
        [96, 208]), t([16, 48]), t([64])]), "inception_4a/"))
    model.add(inception_layer_v1(512, t([t([160]), t(
        [112, 224]), t([24, 64]), t([64])]), "inception_4b/"))
    model.add(inception_layer_v1(512, t([t([128]), t(
        [128, 256]), t([24, 64]), t([64])]), "inception_4c/"))
    model.add(inception_layer_v1(512, t([t([112]), t(
        [144, 288]), t([32, 64]), t([64])]), "inception_4d/"))
    model.add(inception_layer_v1(528, t([t([256]), t(
        [160, 320]), t([32, 128]), t([128])]), "inception_4e/"))
    model.add(SpatialMaxPooling(3, 3, 2, 2, to_ceil=True))
    model.add(inception_layer_v1(832, t([t([256]), t(
        [160, 320]), t([32, 128]), t([128])]), "inception_5a/"))
    model.add(inception_layer_v1(832, t([t([384]), t(
        [192, 384]), t([48, 128]), t([128])]), "inception_5b/"))
    model.add(SpatialAveragePooling(7, 7, 1, 1).set_name("pool5/7x7_s1"))
    if has_dropout:
        model.add(Dropout(0.4).set_name("pool5/drop_7x7_s1"))
    model.add(View([1024], num_input_dims=3))
    model.add(Linear(1024, class_num)
              .set_init_method(weight_init_method=Xavier(), bias_init_method=Zeros())
              .set_name("loss3/classifier"))
    model.add(LogSoftMax().set_name("loss3/loss3"))
    model.reset()
    return model


if __name__ == "__main__":
    # parse options
    parser = config_option_parser()
    (options, args) = parser.parse_args(sys.argv)

    if not options.learningRate:
        parser.error("-l --learningRate is a mandatory opt")
    if not options.batchSize:
        parser.error("-b --batchSize is a mandatory opt")

    # init
    sc = init_nncontext("inception v1")

    image_size = 224  # create dataset
    train_transformer = ChainedPreprocessing([ImagePixelBytesToMat(),
                                              ImageResize(256, 256),
                                              ImageRandomCrop(image_size, image_size),
                                              ImageRandomPreprocessing(ImageHFlip(), 0.5),
                                              ImageChannelNormalize.for_3_channels(123.0, 117.0, 104.0),
                                              ImageMatToTensor(format="NCHW", to_RGB=False),
                                              ImageSetToSample(input_keys=["imageTensor"],
                                                               target_keys=["label"])
                                              ])

    raw_train_data = get_inception_data(options.folder, sc, "train")
    train_data = FeatureSet.image_frame(raw_train_data).transform(train_transformer)

    val_transformer = ChainedPreprocessing([ImagePixelBytesToMat(),
                                            ImageResize(256, 256),
                                            ImageCenterCrop(image_size, image_size),
                                            ImageChannelNormalize.for_3_channels(123.0, 117.0, 104.0),
                                            ImageMatToTensor(format="NCHW", to_RGB=False),
                                            ImageSetToSample(input_keys=["imageTensor"],
                                                             target_keys=["label"])
                                            ])

    raw_val_data = get_inception_data(options.folder, sc, "val")
    val_data = FeatureSet.image_frame(raw_val_data).transform(val_transformer)

    # build model
    if options.model != "":
        # load model snapshot
        inception_model = Model.load(options.model)
    else:
        inception_model = inception_v1_no_aux_classifier(options.classNum)

    # set optimization method
    iterationPerEpoch = int(ceil(float(1281167) / options.batchSize))
    if options.maxEpoch:
        maxIteration = iterationPerEpoch * options.maxEpoch
    else:
        maxIteration = options.maxIteration
    warmup_iteration = options.warmupEpoch * iterationPerEpoch
    if options.state != "":
        # load state snapshot
        optim = OptimMethod.load(options.state)
    else:
        if warmup_iteration == 0:
            warmupDelta = 0.0
        else:
            if options.maxLr:
                maxlr = options.maxLr
            else:
                maxlr = options.learningRate
            warmupDelta = (maxlr - options.learningRate)/warmup_iteration
        polyIteration = maxIteration - warmup_iteration
        lrSchedule = SequentialSchedule(iterationPerEpoch)
        lrSchedule.add(Warmup(warmupDelta), warmup_iteration)
        lrSchedule.add(Poly(0.5, maxIteration), polyIteration)
        optim = SGD(learningrate=options.learningRate, learningrate_decay=0.0,
                    weightdecay=options.weightDecay,
                    momentum=0.9, dampening=0.0, nesterov=False,
                    leaningrate_schedule=lrSchedule)

    # create triggers
    if options.maxEpoch:
        checkpoint_trigger = EveryEpoch()
        test_trigger = EveryEpoch()
        end_trigger = MaxEpoch(options.maxEpoch)
    else:
        checkpoint_trigger = SeveralIteration(options.checkpointIteration)
        test_trigger = SeveralIteration(options.checkpointIteration)
        end_trigger = MaxIteration(options.maxIteration)

    timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Estimator
    estimator = Estimator(inception_model, optim_methods=optim, model_dir=options.checkpoint)

    if options.gradientMin and options.gradientMax:
        estimator.set_constant_gradient_clipping(options.gradientMin, options.gradientMax)

    if options.gradientL2NormThreshold:
        estimator.set_l2_norm_gradient_clipping(options.gradientL2NormThreshold)

    estimator.train_imagefeature(train_set=train_data,
                                 criterion=ClassNLLCriterion(),
                                 end_trigger=end_trigger,
                                 checkpoint_trigger=checkpoint_trigger,
                                 validation_set=val_data,
                                 validation_method=[Top1Accuracy(), Top5Accuracy()],
                                 batch_size=options.batchSize)

    inception_model.saveModel("/tmp/inception/model.bigdl", "/tmp/inception/model.bin", True)

    sc.stop()
