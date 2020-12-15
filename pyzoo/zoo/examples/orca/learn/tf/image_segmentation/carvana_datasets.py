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
import tensorflow_datasets.public_api as tfds
import tensorflow as tf
import os


class Carvana(tfds.core.GeneratorBasedBuilder):
    """Short description of my dataset."""

    VERSION = tfds.core.Version('0.1.0')
    
    MANUAL_DOWNLOAD_INSTRUCTIONS = """\
    Download train.zip and train_masks.zip to {data_dir}/downloads/manual/carvana/,
    {data_dir} defaults to ~/tensorflow_datasets/
    """

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            # tfds.features.FeatureConnectors
            features=tfds.features.FeaturesDict({
                "image_description": tfds.features.Text(),
                "image": tfds.features.Image(),
                "mask": tfds.features.Image(shape=(None, None, None, 3)),
            }),
            supervised_keys=("image", "mask"))

    def _split_generators(self, dl_manager):
        # Download source data
        train_path = os.path.join(dl_manager.manual_dir, 'train.zip')
        train_mask_path = os.path.join(dl_manager.manual_dir, 'train_masks.zip')
        extracted_train_path = dl_manager.extract(train_path)
        extracted_train_mask_path = dl_manager.extract(train_mask_path)

        # Specify the splits
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "train_dir_path": os.path.join(extracted_train_path, "train"),
                    "train_mask_dir_path": os.path.join(extracted_train_mask_path, "train_masks"),
                },
            ),
        ]

    def _generate_examples(self, train_dir_path, train_mask_dir_path):
        # Read the input data out of the source files
        data = []
        import tensorflow as tf
        for image_file in tf.io.gfile.listdir(train_dir_path):
            image_id = image_file[:-4] # xxx.jpg
            mask_file = f"{image_id}_mask.gif"
            data.append((image_id, image_file, mask_file))

        # And yield examples as feature dictionaries
        for image_id, image, mask in data:
            yield image_id, {
                "image_description": image_id,
                "image": os.path.join(train_dir_path, image),
                "mask": os.path.join(train_mask_dir_path, mask),
            }
            
if __name__ == "__main__":
    dataset_builder = Carvana()
    dataset_builder.download_and_prepare()