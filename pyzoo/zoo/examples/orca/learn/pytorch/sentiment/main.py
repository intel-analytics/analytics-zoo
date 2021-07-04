
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
# ==============================================================================
# Most of the pytorch code is adapted from
# https://github.com/prakashpandey9/Text-Classification-Pytorch

from __future__ import print_function
import argparse
import numpy as np
from os.path import exists
from os import makedirs
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe
from torchtext import data, datasets
from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.learn.pytorch import Estimator
from zoo.orca.learn.metrics import Accuracy


parser = argparse.ArgumentParser(description='PyTorch Sentiment Example')
parser.add_argument('--cluster_mode', type=str, default="local",
                    help='The cluster mode, local or yarn')
args = parser.parse_args()
if args.cluster_mode == "local":
    init_orca_context(memory="4g")
elif args.cluster_mode == "yarn":
    init_orca_context(
        cluster_mode="yarn-client", num_nodes=2, driver_memory="4g",
        conf={"spark.rpc.message.maxSize": "1024",
              "spark.task.maxFailures": "1",
              "spark.driver.extraJavaOptions": "-Dbigdl.failure.retryTimes=1"})


class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
        super(LSTMClassifier, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as
        the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which
        we will use to create our word_embedding look-up table

        """

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        self.word_embeddings.weight = nn.Parameter(weights,
                                                   requires_grad=False)
        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    def forward(self, input_sentence, batch_size=None):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction
        on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for positive & negative class
        which receives its input as the final_hidden_state of the LSTM
        final_output.shape = (batch_size, output_size)

        """

        ''' Here we will map all the indexes present in the input sequence to
        the corresponding word vector using our pre-trained word_embedddins.'''
        input = self.word_embeddings(
            input_sentence)
        input = input.permute(1, 0, 2)
        if batch_size is None:
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        final_output = self.label(final_hidden_state[-1])
        return final_output


def text_label_creator():
    # load the dataset and build the vocabulary
    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True,
                      include_lengths=True, batch_first=True, fix_length=200)
    LABEL = data.LabelField()
    train_dataset, _ = datasets.IMDB.splits(TEXT, LABEL)
    TEXT.build_vocab(train_dataset, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train_dataset)
    return TEXT, LABEL


def model_creator(config):
    TEXT = config.get("TEXT_LABEL")[0]
    word_embeddings = TEXT.vocab.vectors
    vocab_size = len(TEXT.vocab)
    batch_size = 32
    output_size = 2
    hidden_size = 256
    embedding_length = 300
    model = LSTMClassifier(batch_size, output_size, hidden_size,
                           vocab_size, embedding_length, word_embeddings)
    return model


def optim_creator(model, config):
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    return optim


class MyCollator(object):
    def __init__(self, TEXT):
        self.TEXT = TEXT

    def __call__(self, data):
        label_list = [int(d.label == 'pos') for d in data]
        label_tensor = torch.LongTensor(label_list)
        txt_list = [d.text for d in data]
        txt_tensor = self.TEXT.process(txt_list)[0]
        return txt_tensor, label_tensor


def train_loader_creator(config, batch_size):
    TEXT_LABEL = config.get("TEXT_LABEL")
    TEXT = TEXT_LABEL[0]
    LABEL = TEXT_LABEL[1]
    train_dataset, _ = datasets.IMDB.splits(TEXT, LABEL)
    train_dataset, _ = train_dataset.split(split_ratio=0.8)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  collate_fn=MyCollator(TEXT), drop_last=True, shuffle=True)
    return train_dataloader


def test_loader_creator(config, batch_size):
    TEXT_LABEL = config.get("TEXT_LABEL")
    TEXT = TEXT_LABEL[0]
    LABEL = TEXT_LABEL[1]
    _, test_dataset = datasets.IMDB.splits(TEXT, LABEL)
    test_dataset, _ = test_dataset.split(split_ratio=0.8)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 collate_fn=MyCollator(TEXT), drop_last=True, shuffle=True)
    return test_dataloader


model_dir = "model_save"
if not exists(model_dir):
    makedirs(model_dir)
model_save_path = model_dir+"/model"
criterion = nn.CrossEntropyLoss()
batch_size = 32

orca_estimator = Estimator.from_torch(model=model_creator,
                                      optimizer=optim_creator,
                                      loss=criterion,
                                      workers_per_node=2,
                                      metrics=[Accuracy()],
                                      backend="torch_distributed",
                                      config={"lr": 2e-5,
                                              "TEXT_LABEL": text_label_creator()})
orca_estimator.fit(data=train_loader_creator, epochs=5, batch_size=batch_size)
model = orca_estimator.get_model()
torch.save(model.state_dict(), model_save_path)
res = orca_estimator.evaluate(data=test_loader_creator)
for r in res:
    print(r, ":", res[r])
stop_orca_context()

# start testing
print("***Finish training, start testing***")
config = {"TEXT_LABEL": text_label_creator()}
model = model_creator(config)
model.load_state_dict(torch.load(model_save_path))
model.eval()
test_sen1 = '''This is one of the best creation of Nolan. I can say, it's his magnum opus.
                Loved the soundtrack and especially those creative dialogues.'''
test_sen2 = '''Ohh, such a ridiculous movie. Not gonna recommend it to anyone.
                Complete waste of time and money.'''
test_sen_set = [test_sen1, test_sen2]
TEXT = config.get("TEXT_LABEL")[0]
for test_sen in test_sen_set:
    print(test_sen)
    test_sen = TEXT.preprocess(test_sen)
    test_sen = [[TEXT.vocab.stoi[x] for x in test_sen]]
    test_sen = np.asarray(test_sen)
    test_sen = torch.LongTensor(test_sen)
    test_tensor = Variable(test_sen)
    output = model(test_tensor, 1)
    out = F.softmax(output, 1)
    if (torch.argmax(out[0]) == 1):
        print("Sentiment: Positive")
    else:
        print("Sentiment: Negative")
