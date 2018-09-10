import argparse
from moviepy.editor import *

from zoo.common.nncontext import *
from zoo.feature.image import *
from zoo.feature.common import ChainedPreprocessing
from zoo.models.image.common.image_config import ImageConfigure
from zoo.models.image.objectdetection import *

parser = argparse.ArgumentParser()
parser.add_argument('model_path', help="Path where the model is stored")

sc = init_nncontext(create_spark_conf().setAppName("Detect Messi Example with fasterrcnn"))

args = parser.parse_args()
model=ObjectDetector.load_model(args.model_path)
model.evaluate()

path = "messi_clip.mp4"
myclip = VideoFileClip(path)

video_rdd = sc.parallelize(myclip.iter_frames(fps=5))
image_set = DistributedImageSet(video_rdd)

label_map = {0: '__background__', 1: 'messi'}
preprocess = ChainedPreprocessing([ImageAspectScale(600, 1),
                                   ImageChannelNormalize(122.7717, 115.9465, 102.9801),
                                    ImageMatToTensor(), ImInfo(), DummyGT(),
                                   ImageSetToSample(["imageTensor", "ImInfo", "DummyGT"])])
postprocess = DecodeOutput()

config = ImageConfigure(preprocess, postprocess, 1, label_map)

output = model.predict_image_set(image_set, config)

visualizer = Visualizer(config.label_map())
visualized = visualizer(output).get_image(to_chw=False).collect()

clip = ImageSequenceClip(visualized, fps=5)

clip.write_gif("messi.gif")
