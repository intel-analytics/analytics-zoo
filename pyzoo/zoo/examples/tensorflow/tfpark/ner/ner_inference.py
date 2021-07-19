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

from __future__ import division, print_function, unicode_literals, absolute_import

import argparse
import pickle

import re
import numpy as np
import tensorflow as tf

from nlp_architect.utils.generic import pad_sentences
from nlp_architect.utils.io import validate_existing_filepath
from zoo.common.nncontext import init_nncontext
from zoo.pipeline.api.net import TFDataset
from zoo.tfpark.text import NER


def read_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=validate_existing_filepath, required=True,
                        help='The path to model weights')
    parser.add_argument('--model_info_path', type=validate_existing_filepath, required=True,
                        help='The path to model topology')
    parser.add_argument('--seq_length', type=int, default=30, help='The maximum sequence length of inputs')
    parser.add_argument('--input_path', help='The path to the input txt file if any. Each line should be a sentence')
    input_args = parser.parse_args()
    return input_args


def preprocess(text, w_vocab, c_vocab, word_length):
    tokens = process_text(text)
    return vectorize(tokens, w_vocab, c_vocab, word_length)


def process_text(doc):
    return re.sub("[^a-zA-Z]", " ", doc).strip().split()


# Word to index and char to index with padding. 1 means OOV and 0 means pad.
# Words are pre-trained in lower case while chars differ lower and upper cases.
# For a single input, seq_length doesn't need to be fixed;
# while for batch input, each input should be of the same seq_length.
# seq_length needs to the max length of all inputs; otherwise words will be truncated.
def vectorize(doc, w_vocab, c_vocab, word_length, crf_mode="reg", seq_length=None):
    words = np.asarray([w_vocab[w.lower()] if w.lower() in w_vocab else 1 for w in doc])
    sentence_chars = []
    for w in doc:
        word_chars = []
        for c in w:
            if c in c_vocab:
                _cid = c_vocab[c]
            else:
                _cid = 1
            word_chars.append(_cid)
        sentence_chars.append(word_chars)
    if seq_length:
        words = pad_sentences(words.reshape(1, -1), seq_length).reshape(-1)
        for i in range(0, seq_length - len(doc)):
            sentence_chars.append([0])
    sentence_chars = pad_sentences(sentence_chars, word_length)
    if crf_mode == "reg":
        return [words, sentence_chars]
    else:  # "pad"
        assert seq_length, "seq_length can't be None in CRF pad mode"
        return [words, sentence_chars, np.array([seq_length])]


if __name__ == '__main__':
    args = read_input_args()
    # Load model weights and topology information
    with open(args.model_info_path, 'rb') as fp:
        model_info = pickle.load(fp)
    assert model_info is not None, 'No model topology information loaded'
    model = NER.load_model(args.model_path)
    word_vocab = model_info['word_vocab']
    y_vocab = {v: k for k, v in model_info['y_vocab'].items()}
    char_vocab = model_info['char_vocab']
    word_length = model.labor.word_length
    crf_mode = model.labor.crf_mode
    seq_length = args.seq_length

    sc = init_nncontext("NER distributed inference example")
    word_vocab_broad = sc.broadcast(word_vocab)
    char_vocab_broad = sc.broadcast(char_vocab)

    # Read texts as RDD of String
    if not args.input_path:
        texts = ["John is planning a visit to London on October",
                 "even though Intel is a big organization, purchasing Mobileye last year had a huge positive impact"]
        text_rdd = sc.parallelize(texts)
    else:
        text_rdd = sc.textFile(args.input_path)

    # Preprocessing
    tokens_rdd = text_rdd.map(lambda x: process_text(x))
    indices_rdd = tokens_rdd.map(lambda tokens: vectorize(tokens, word_vocab_broad.value,
                                                          char_vocab_broad.value, word_length, crf_mode, seq_length))
    tensor_names = ["word_indices", "char_indices"]
    tensor_shapes = [[seq_length], [seq_length, word_length]]
    tensor_types = [tf.int32, tf.int32]
    if model.labor.crf_mode == 'pad':
        tensor_names = tensor_names + ["seq_length"]
        tensor_shapes = tensor_shapes + [[1]]
        tensor_types = tensor_types + [tf.int32]
    predict_dataset = TFDataset.from_rdd(indices_rdd,
                                         names=tensor_names,
                                         shapes=tensor_shapes,
                                         types=tensor_types,
                                         batch_per_thread=4)
    result_rdd = model.predict(predict_dataset, distributed=True)

    # Print the result of the first five texts for illustration
    # Can further do postprocessing to combine and remove prefix B-&I- in entities.
    for (tokens, results) in tokens_rdd.zip(result_rdd).take(5):
        ners = [y_vocab.get(n, None) for n in results.argmax(1)]
        for t, n in zip(tokens, ners):
            print('{}\t{}\t'.format(t, n))
        print()
