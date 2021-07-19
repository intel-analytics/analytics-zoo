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

import time
from optparse import OptionParser

import tensorflow as tf
from zoo.common.nncontext import *
from zoo.tfpark.text.estimator import BERTClassifier, bert_input_fn
from zoo.pipeline.api.keras.optimizers import AdamWeightDecay
from bert import tokenization


# Copy code from BERT run_classifier.py since import run_classifier will have error in imports such as
# import modeling

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      import csv
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[3])
      text_b = tokenization.convert_to_unicode(line[4])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features


def feature_to_input(feature):
    res = dict()
    res["input_ids"] = np.array(feature.input_ids)
    res["input_mask"] = np.array(feature.input_mask)
    res["token_type_ids"] = np.array(feature.segment_ids)
    return res, np.array(feature.label_id)


def generate_input_rdd(examples, label_list, max_seq_length, tokenizer, type="train"):
    features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer)
    features = [feature_to_input(feature) for feature in features]
    if type == "test":
        return sc.parallelize(features).map(lambda x: x[0])
    else:
        return sc.parallelize(features)


if __name__ == '__main__':
    start_time = time.time()
    parser = OptionParser()
    parser.add_option("--bert_base_dir", dest="bert_base_dir")
    parser.add_option("--data_dir", dest="data_dir")
    parser.add_option("--output_dir", dest="output_dir")
    parser.add_option("--batch_size", dest="batch_size", type=int, default=32)
    parser.add_option("--max_seq_length", dest="max_seq_length", type=int, default=128)
    parser.add_option("-e", "--nb_epoch", dest="nb_epoch", type=int, default=3)
    parser.add_option("-l", "--learning_rate", dest="learning_rate", type=float, default=2e-5)
    parser.add_option("--do_train", dest="do_train", type=int, default=1)
    parser.add_option("--do_eval", dest="do_eval", type=int, default=1)
    parser.add_option("--do_predict", dest="do_predict", type=int, default=1)

    (options, args) = parser.parse_args(sys.argv)
    sc = init_nncontext("BERT MRPC Classification Example")

    processor = MrpcProcessor()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(os.path.join(options.bert_base_dir, "vocab.txt"))
    estimator = BERTClassifier(len(label_list), bert_config_file=os.path.join(options.bert_base_dir, "bert_config.json"),
                               init_checkpoint=os.path.join(options.bert_base_dir, "bert_model.ckpt"),
                               # optimizer=tf.train.AdamOptimizer(options.learning_rate),
                               model_dir=options.output_dir)

    # Training
    if options.do_train:
        train_examples = processor.get_train_examples(options.data_dir)
        steps = len(train_examples) * options.nb_epoch // options.batch_size
        optimizer = AdamWeightDecay(lr=options.learning_rate, warmup_portion=0.1, total=steps)
        estimator.set_optimizer(optimizer)
        train_rdd = generate_input_rdd(train_examples, label_list, options.max_seq_length, tokenizer, "train")
        train_input_fn = bert_input_fn(train_rdd, options.max_seq_length, options.batch_size)
        train_start_time = time.time()
        estimator.train(train_input_fn, steps=steps)
        train_end_time = time.time()
        print("Train time: %s minutes" % ((train_end_time - train_start_time) / 60))

    # Evaluation
    if options.do_eval:
        eval_examples = processor.get_dev_examples(options.data_dir)
        eval_rdd = generate_input_rdd(eval_examples, label_list, options.max_seq_length, tokenizer, "eval")
        eval_input_fn = bert_input_fn(eval_rdd, options.max_seq_length, options.batch_size)
        eval_start_time = time.time()
        result = estimator.evaluate(eval_input_fn, eval_methods=["acc"])
        print(result)
        eval_end_time = time.time()
        print("Eval time: %s minutes" % ((eval_end_time - eval_start_time) / 60))

    # Inference
    if options.do_predict:
        test_examples = processor.get_test_examples(options.data_dir)
        test_rdd = generate_input_rdd(test_examples, label_list, options.max_seq_length, tokenizer, "test")
        test_input_fn = bert_input_fn(test_rdd, options.max_seq_length, options.batch_size)
        predictions = estimator.predict(test_input_fn)
        pred_start_time = time.time()
        # predictions.collect()
        pred_end_time = time.time()
        print("Inference time: %s minutes" % ((pred_end_time - pred_start_time) / 60))
        print("Inference throughput: %s records/s" % (len(test_examples) / (pred_end_time - pred_start_time)))
        for prediction in predictions.take(5):
            print(prediction)

    end_time = time.time()
    print("Time elapsed: %s minutes" % ((end_time - start_time) / 60))
    print("Finished")
