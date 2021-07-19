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
import pickle
from optparse import OptionParser

from zoo.common.nncontext import *
from zoo.tfpark.text.estimator import BERTNER, bert_input_fn
from zoo.pipeline.api.keras.optimizers import AdamWeightDecay
from bert import tokenization


# Copy code from BERT BERT_NER.py https://github.com/kyzhouhzau/BERT-NER

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text = text
    self.label = label

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               mask,
               segment_ids,
               label_ids,
               is_real_example=True):
    self.input_ids = input_ids
    self.mask = mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids
    self.is_real_example = is_real_example

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls,input_file):
        """Read a BIO data!"""
        rf = open(input_file,'r')
        lines = [];words = [];labels = []
        for line in rf:
            word = line.strip().split(' ')[0]
            label = line.strip().split(' ')[-1]
            # here we dont do "DOCSTART" check
            if len(line.strip())==0 and words[-1] == '.':
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append((l,w))
                words=[]
                labels = []
            words.append(word)
            labels.append(label)
        rf.close()
        return lines

class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test"
        )


    def get_labels(self):
        """
        here "X" used to represent "##eer","##soo" and so on!
        "[PAD]" for padding
        :return:
        """
        return ["[PAD]","B-MISC", "I-MISC", "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X","[CLS]","[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            texts = tokenization.convert_to_unicode(line[1])
            labels = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=texts, label=labels))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
    """
    :param ex_index: example num
    :param example:
    :param label_list: all labels
    :param max_seq_length:
    :param tokenizer: WordPiece tokenization
    :param mode:
    :return: feature

    IN this part we should rebuild input sentences to the following format.
    example:[Jim,Hen,##son,was,a,puppet,##eer]
    labels: [I-PER,I-PER,X,O,O,O,X]

    """
    label_map = {}
    #here start with zero this means that "[PAD]" is zero
    for (i,label) in enumerate(label_list):
        label_map[label] = i
    with open(options.output_dir + "/label2id.pkl",'wb') as w:
        pickle.dump(label_map,w)
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i,(word,label) in enumerate(zip(textlist,labellist)):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for i,_ in enumerate(token):
            if i==0:
                labels.append(label)
            else:
                labels.append("X")
    # only Account for [CLS] with "- 1".
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 1)]
        labels = labels[0:(max_seq_length - 1)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    # after that we don't add "[SEP]" because we want a sentence don't have
    # stop tag, because i think its not very necessary.
    # or if add "[SEP]" the model even will cause problem, special the crf layer was used.
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    mask = [1]*len(input_ids)
    #use zero to padding and you should
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append("[PAD]")
    assert len(input_ids) == max_seq_length
    assert len(mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(ntokens) == max_seq_length
    feature = InputFeatures(
        input_ids=input_ids,
        mask=mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
    )
    # we need ntokens because if we do predict it can help us return to original token.
    return feature,ntokens,label_ids


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
      feature, ntokens, label_ids = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)
      features.append(feature)
  return features


def feature_to_input(feature):
    res = dict()
    res["input_ids"] = np.array(feature.input_ids)
    res["input_mask"] = np.array(feature.mask)
    res["token_type_ids"] = np.array(feature.segment_ids)
    return res, np.array(feature.label_ids)


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
    sc = init_nncontext("BERT NER Example")

    processor = NerProcessor()
    label_list = processor.get_labels()
    # Recommended to use cased model for NER
    tokenizer = tokenization.FullTokenizer(os.path.join(options.bert_base_dir, "vocab.txt"), do_lower_case=False)
    estimator = BERTNER(len(label_list),
                        bert_config_file=os.path.join(options.bert_base_dir, "bert_config.json"),
                        init_checkpoint=os.path.join(options.bert_base_dir, "bert_model.ckpt"),
                        model_dir=options.output_dir)

    # Training
    if options.do_train:
        train_examples = processor.get_train_examples(options.data_dir)
        steps = len(train_examples) * options.nb_epoch // options.batch_size
        optimizer = AdamWeightDecay(lr=options.learning_rate, warmup_portion=0.1, total=steps)
        estimator.set_optimizer(optimizer)
        estimator.set_gradient_clipping_by_l2_norm(1.0)
        train_rdd = generate_input_rdd(train_examples, label_list, options.max_seq_length, tokenizer, "train")
        train_input_fn = bert_input_fn(train_rdd, options.max_seq_length, options.batch_size,
                                       label_size=options.max_seq_length)
        train_start_time = time.time()
        estimator.train(train_input_fn, steps=steps)
        train_end_time = time.time()
        print("Train time: %s minutes" % ((train_end_time - train_start_time) / 60))

    # Evaluation
    # Confusion matrix is not supported and thus use sklearn classification_report for evaluation
    if options.do_eval:
        eval_examples = processor.get_dev_examples(options.data_dir)
        eval_rdd = generate_input_rdd(eval_examples, label_list, options.max_seq_length, tokenizer, "eval")
        eval_input_fn = bert_input_fn(eval_rdd, options.max_seq_length, options.batch_size)
        result = estimator.predict(eval_input_fn).zip(eval_rdd).collect()
        predictions = np.concatenate([r[0] for r in result])
        truths = np.concatenate([r[1][1] for r in result])
        mask = np.concatenate([r[1][0]["input_mask"] for r in result])
        with open(os.path.join(options.output_dir, "label2id.pkl"), 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
        sorted_ids = sorted(id2label.keys())
        labels = [id2label[id] for id in sorted_ids]
        from sklearn.metrics import classification_report
        print(classification_report(truths, predictions, sample_weight=mask,
                                    labels=sorted_ids, target_names=labels))

    # Inference
    if options.do_predict:
        test_examples = processor.get_test_examples(options.data_dir)
        test_rdd = generate_input_rdd(test_examples, label_list, options.max_seq_length, tokenizer, "test")
        test_input_fn = bert_input_fn(test_rdd, options.max_seq_length, options.batch_size)
        predictions = estimator.predict(test_input_fn)
        pred_start_time = time.time()
        predictions.collect()
        pred_end_time = time.time()
        print("Inference time: %s minutes" % ((pred_end_time - pred_start_time) / 60))
        print("Inference throughput: %s records/s" % (len(test_examples) / (pred_end_time - pred_start_time)))
        for prediction in predictions.take(5):
            print(prediction)

    end_time = time.time()
    print("Time elapsed: %s minutes" % ((end_time - start_time) / 60))
    print("Finished")
