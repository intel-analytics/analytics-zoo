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

import shutil
import errno

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pytest

from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.pipeline.api.net.torch_net import TorchNet
from zoo.pipeline.api.net.torch_criterion import TorchCriterion
from zoo.pipeline.nnframes import *
from zoo.common.nncontext import *

from pyspark.ml.linalg import Vectors


class TestPytorch(ZooTestCase):

    def setup_method(self, method):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        sparkConf = init_spark_conf().setMaster("local[1]").setAppName("TestPytorch")
        self.sc = init_nncontext(sparkConf)
        self.sqlContext = SQLContext(self.sc)
        assert(self.sc.appName == "TestPytorch")
        if self.sc.version.startswith("2"):
            from pyspark.sql import SparkSession
            spark = SparkSession \
                .builder \
                .getOrCreate()

    def teardown_method(self, method):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        self.sc.stop()

    def test_torchnet_constructor(self):
        # two inputs test
        class TwoInputModel(nn.Module):
            def __init__(self):
                super(TwoInputModel, self).__init__()
                self.dense1 = nn.Linear(2, 2)
                self.dense2 = nn.Linear(3, 1)

            def forward(self, x1, x2):
                x1 = self.dense1(x1)
                x2 = self.dense2(x2)
                return x1, x2

        TorchNet.from_pytorch(TwoInputModel(), (torch.ones(2, 2), torch.ones(2, 3)))
        TorchNet.from_pytorch(TwoInputModel(), ([2, 2], [2, 3]))
        TorchNet.from_pytorch(TwoInputModel(), [torch.ones(2, 2), torch.ones(2, 3)])
        TorchNet.from_pytorch(TwoInputModel(), [[2, 2], [2, 3]])

        # one input
        input = [[0.5, 1.], [-0.3, 1.2]]
        torch_input = torch.tensor(input)
        model = nn.Linear(2, 1)
        TorchNet.from_pytorch(model, torch_input)
        TorchNet.from_pytorch(model, [1, 2])

    def test_torchcriterion_constructor(self):
        # two inputs test
        criterion = nn.MSELoss()

        def lossFunc(input, label):
            loss1 = criterion(input[0], label[0])
            loss2 = criterion(input[1], label[1])
            loss = loss1 + 0.4 * loss2
            return loss

        TorchCriterion.from_pytorch(lossFunc,
                                    (torch.ones(2, 2), torch.ones(2, 3)),
                                    (torch.ones(2, 2), torch.ones(2, 3)))
        TorchCriterion.from_pytorch(lossFunc, ([2, 2], [2, 3]), ([2, 2], [2, 3]))
        TorchCriterion.from_pytorch(lossFunc,
                                    [torch.ones(2, 2), torch.ones(2, 3)],
                                    [torch.ones(2, 2), torch.ones(2, 3)])
        TorchCriterion.from_pytorch(lossFunc, [[2, 2], [2, 3]], [[2, 2], [2, 3]])

        # one inputs test
        TorchCriterion.from_pytorch(criterion, [2, 1], [2, 1])
        TorchCriterion.from_pytorch(criterion, torch.ones(2, 2), torch.ones(2, 2))

    def test_torch_net_predict_resnet(self):
        try:
            model = torchvision.models.resnet18(pretrained=True).eval()
        except:
            # try again to work around connection error
            import time
            import random
            time.sleep(random.randint(0, 20))
            model = torchvision.models.resnet18(pretrained=True).eval()
        net = TorchNet.from_pytorch(model, [1, 3, 224, 224])

        dummpy_input = torch.ones(1, 3, 224, 224)
        result = net.predict(dummpy_input.numpy()).collect()
        assert np.allclose(result[0][0:5].tolist(),
                           np.asarray(
                               [-0.03913354128599167, 0.11446280777454376, -1.7967549562454224,
                                -1.2342952489852905, -0.819004476070404]))

    def test_linear_gradient_match(self):
        input = [[0.5, 1.], [-0.3, 1.2]]
        label = [[0.6], [-0.9]]
        torch_input = torch.tensor(input)
        torch_label = torch.tensor(label)

        model = nn.Linear(2, 1)
        criterion = nn.MSELoss()

        torch_output = model.forward(torch_input)
        torch_loss = criterion.forward(torch_output, torch_label)
        torch_loss.backward()
        torch_grad = model.weight.grad.tolist()[0] + model.bias.grad.tolist()

        # AZ part
        az_net = TorchNet.from_pytorch(model, [1, 2])
        az_criterion = TorchCriterion.from_pytorch(criterion, [1, 1], [1, 1])

        az_input = np.array(input)
        az_label = np.array(label)

        az_output = az_net.forward(az_input)
        az_loss_output = az_criterion.forward(az_output, az_label)
        az_loss_backward = az_criterion.backward(az_output, az_label)
        az_model_backward = az_net.backward(az_input, az_loss_backward)

        az_grad = list(az_net.parameters().values())[0]['gradWeight']

        assert np.allclose(torch_loss.tolist(), az_loss_output)
        assert np.allclose(torch_grad, az_grad.tolist())

    def test_conv2D_gradient_match(self):
        class SimpleTorchModel(nn.Module):
            def __init__(self):
                super(SimpleTorchModel, self).__init__()
                self.dense1 = nn.Linear(2, 48)
                self.conv1 = nn.Conv2d(3, 2, 2)
                self.dense2 = nn.Linear(2, 1)

            def forward(self, x):
                x = self.dense1(x)
                x = x.view(-1, 3, 4, 4)
                x = torch.relu(self.conv1(x))
                x = F.max_pool2d(x, 2)
                x = x.view(x.size(0), -1)
                x = torch.sigmoid(self.dense2(x))
                return x

        input = [[1., -0.5], [0.5, -1.]]
        label = [[1., -0.5]]
        torch_input = torch.tensor(input)
        torch_label = torch.tensor(label)

        torch_model = SimpleTorchModel()
        torch_criterion = nn.MSELoss()

        torch_output = torch_model.forward(torch_input)
        torch_loss = torch_criterion.forward(torch_output, torch_label)
        torch_loss.backward()
        torch_grad = torch_model.dense1.weight.grad.flatten().tolist() + \
            torch_model.dense1.bias.grad.flatten().tolist() + \
            torch_model.conv1.weight.grad.flatten().tolist() + \
            torch_model.conv1.bias.grad.flatten().tolist() + \
            torch_model.dense2.weight.grad.flatten().tolist() + \
            torch_model.dense2.bias.grad.flatten().tolist()

        # AZ part
        az_net = TorchNet.from_pytorch(torch_model, [1, 2])
        az_criterion = TorchCriterion.from_pytorch(torch_criterion.forward, [1, 1], [1, 1])

        az_input = np.array(input)
        az_label = np.array(label)
        az_output = az_net.forward(np.array(input))
        az_loss_output = az_criterion.forward(az_output, az_label)
        az_loss_backward = az_criterion.backward(az_output, az_label)
        az_model_backward = az_net.backward(az_input, az_loss_backward)

        az_grad = list(az_net.parameters().values())[0]['gradWeight']

        assert np.allclose(torch_loss.tolist(), az_loss_output)
        assert np.allclose(torch_grad, az_grad.tolist())

    def test_cross_entrophy_match(self):
        input = [[0.5, 1.], [-0.3, 1.2]]
        label = [3, 6]
        torch_input = torch.tensor(input)
        torch_label = torch.tensor(label).long()

        model = nn.Linear(2, 10)
        criterion = nn.CrossEntropyLoss()

        def lossFunc(input, target):
            return criterion.forward(input, target.flatten().long())

        torch_output = model.forward(torch_input)
        torch_loss = criterion.forward(torch_output, torch_label)
        torch_loss.backward()
        torch_grad = model.weight.grad.flatten().tolist() + model.bias.grad.tolist()

        # AZ part
        az_net = TorchNet.from_pytorch(model, [1, 2])
        az_criterion = TorchCriterion.from_pytorch(lossFunc, [1, 10], [1, 1])

        az_input = np.array(input)
        az_label = np.array(label)

        az_output = az_net.forward(az_input)
        az_loss_output = az_criterion.forward(az_output, az_label)
        az_loss_backward = az_criterion.backward(az_output, az_label)
        az_model_backward = az_net.backward(az_input, az_loss_backward)

        az_grad = list(az_net.parameters().values())[0]['gradWeight']

        assert np.allclose(torch_loss.tolist(), az_loss_output)
        assert np.allclose(torch_grad, az_grad.tolist())

    def test_Lenet_gradient_match(self):
        class LeNet(nn.Module):
            def __init__(self):
                super(LeNet, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5, 1)
                self.conv2 = nn.Conv2d(20, 50, 5, 1)
                self.fc1 = nn.Linear(4 * 4 * 50, 500)
                self.fc2 = nn.Linear(500, 10)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.max_pool2d(x, 2, 2)
                x = F.relu(self.conv2(x))
                x = F.max_pool2d(x, 2, 2)
                x = x.view(-1, 4 * 4 * 50)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return F.log_softmax(x, dim=1)

        input = np.random.rand(2, 1, 28, 28)
        label = [7, 3]
        torch_input = torch.tensor(input).float()
        torch_label = torch.tensor(label).long()

        torch_model = LeNet()
        torch_criterion = nn.CrossEntropyLoss()

        torch_output = torch_model.forward(torch_input)
        torch_loss = torch_criterion.forward(torch_output, torch_label)
        torch_loss.backward()
        torch_grad = torch_model.conv1.weight.grad.flatten().tolist() + \
            torch_model.conv1.bias.grad.flatten().tolist() + \
            torch_model.conv2.weight.grad.flatten().tolist() + \
            torch_model.conv2.bias.grad.flatten().tolist() + \
            torch_model.fc1.weight.grad.flatten().tolist() + \
            torch_model.fc1.bias.grad.flatten().tolist() + \
            torch_model.fc2.weight.grad.flatten().tolist() + \
            torch_model.fc2.bias.grad.flatten().tolist()

        # AZ part
        az_net = TorchNet.from_pytorch(torch_model, [1, 1, 28, 28])

        def lossFunc(input, target):
            return torch_criterion.forward(input, target.flatten().long())

        az_criterion = TorchCriterion.from_pytorch(lossFunc, [1, 10], [1, 1])

        az_input = np.array(input)
        az_label = np.array(label)
        az_output = az_net.forward(np.array(input))
        az_loss_output = az_criterion.forward(az_output, az_label)
        az_loss_backward = az_criterion.backward(az_output, az_label)
        az_model_backward = az_net.backward(az_input, az_loss_backward)

        az_grad = list(az_net.parameters().values())[0]['gradWeight']

        assert np.allclose(torch_loss.tolist(), az_loss_output)
        assert np.allclose(torch_grad, az_grad.tolist(), atol=1.e-5, rtol=1.e-3)

    def test_model_inference_with_multiple_output(self):
        class TwoOutputModel(nn.Module):
            def __init__(self):
                super(TwoOutputModel, self).__init__()
                self.dense1 = nn.Linear(2, 1)

            def forward(self, x):
                x1 = self.dense1(x)
                return x, x1

        torch_model = TwoOutputModel()
        az_net = TorchNet.from_pytorch(TwoOutputModel(), [1, 2])

        az_input = np.array([[0.5, 1.], [-0.3, 1.2]])
        az_output = az_net.forward(az_input)
        assert (len(az_output) == 2)
        assert (az_output[0].shape == (2, 2))
        assert (az_output[1].shape == (2, 1))

    def test_model_train_with_multiple_output(self):
        class TwoOutputModel(nn.Module):
            def __init__(self):
                super(TwoOutputModel, self).__init__()
                self.dense1 = nn.Linear(2, 1)

            def forward(self, x):
                x1 = self.dense1(x)
                return x, x1

        input = [[0.5, 1.], [-0.3, 1.2]]
        torch_input = torch.tensor(input)
        torch_label = (torch.ones(2, 2), torch.ones(2, 1))

        model = TwoOutputModel()
        criterion = nn.MSELoss()

        def lossFunc(input, label):
            loss1 = criterion(input[0], label[0])
            loss2 = criterion(input[1], label[1])
            loss = loss1 + 0.4 * loss2
            return loss

        torch_output = model.forward(torch_input)
        torch_loss = lossFunc(torch_output, torch_label)
        torch_loss.backward()
        torch_grad = model.dense1.weight.grad.tolist()[0] + model.dense1.bias.grad.tolist()

        az_net = TorchNet.from_pytorch(model, [1, 2])
        az_criterion = TorchCriterion.from_pytorch(
            lossFunc,
            (torch.ones(2, 2), torch.ones(2, 1)),
            (torch.ones(2, 2), torch.ones(2, 1)))

        az_input = np.array(input)
        az_label = [np.ones([2, 2]), np.ones([2, 1])]
        az_output = az_net.forward(az_input)
        az_loss_output = az_criterion.forward(az_output, az_label)
        az_loss_backward = az_criterion.backward(az_output, az_label)
        az_model_backward = az_net.backward(az_input, az_loss_backward)

        az_grad = list(az_net.parameters().values())[0]['gradWeight']

        assert np.allclose(torch_loss.tolist(), az_loss_output)
        assert np.allclose(torch_grad, az_grad.tolist())

    def test_model_train_with_multiple_input(self):
        class TwoInputModel(nn.Module):
            def __init__(self):
                super(TwoInputModel, self).__init__()
                self.dense1 = nn.Linear(2, 2)
                self.dense2 = nn.Linear(2, 1)

            def forward(self, x1, x2):
                x1 = self.dense1(x1)
                x2 = self.dense2(x2)
                return x1, x2

        input = [[0.5, 1.], [-0.3, 1.2]]
        torch_input1 = torch.tensor(input, requires_grad=True)
        torch_input2 = torch.tensor(input, requires_grad=True)
        torch_label = (torch.ones(2, 2), torch.ones(2, 1))

        model = TwoInputModel()
        criterion = nn.MSELoss()

        def lossFunc(input, label):
            loss1 = criterion(input[0], label[0])
            loss2 = criterion(input[1], label[1])
            loss = loss1 + 0.4 * loss2
            return loss

        torch_output = model.forward(torch_input1, torch_input2)
        torch_loss = lossFunc(torch_output, torch_label)
        torch_loss.backward()
        torch_grad = model.dense1.weight.grad.tolist()[0] + \
            model.dense1.weight.grad.tolist()[1] + \
            model.dense1.bias.grad.tolist() + \
            model.dense2.weight.grad.tolist()[0] + \
            model.dense2.bias.grad.tolist()

        az_net = TorchNet.from_pytorch(model, (torch.ones(2, 2), torch.ones(2, 2)))
        az_criterion = TorchCriterion.from_pytorch(
            lossFunc,
            (torch.ones(2, 2), torch.ones(2, 1)),
            (torch.ones(2, 2), torch.ones(2, 1)))

        az_input = [np.array(input), np.array(input)]
        az_label = [np.ones([2, 2]), np.ones([2, 1])]
        az_output = az_net.forward(az_input)
        az_loss_output = az_criterion.forward(az_output, az_label)
        az_loss_backward = az_criterion.backward(az_output, az_label)
        az_model_backward = az_net.backward(az_input, az_loss_backward)

        az_grad = list(az_net.parameters().values())[0]['gradWeight']

        assert np.allclose(torch_loss.tolist(), az_loss_output)
        assert np.allclose(torch_grad, az_grad.tolist())
        assert np.allclose(az_model_backward[0], torch_input1.grad)
        assert np.allclose(az_model_backward[1], torch_input2.grad)

    def test_model_save_load(self):
        class SimpleTorchModel(nn.Module):
            def __init__(self):
                super(SimpleTorchModel, self).__init__()
                self.dense1 = nn.Linear(2, 4)
                self.dense2 = nn.Linear(4, 1)

            def forward(self, x):
                x = self.dense1(x)
                x = torch.sigmoid(self.dense2(x))
                return x

        df = self.sqlContext.createDataFrame(
            [(Vectors.dense([2.0, 1.0]), 1.0),
             (Vectors.dense([1.0, 2.0]), 0.0),
             (Vectors.dense([2.0, 1.0]), 1.0),
             (Vectors.dense([1.0, 2.0]), 0.0)],
            ["features", "label"])

        torch_model = SimpleTorchModel()
        torch_criterion = nn.MSELoss()

        az_model = TorchNet.from_pytorch(torch_model, [1, 2])
        az_criterion = TorchCriterion.from_pytorch(torch_criterion, [1, 1], [1, 1])
        estimator = NNEstimator(az_model, az_criterion) \
            .setBatchSize(4) \
            .setLearningRate(0.01) \
            .setMaxEpoch(10)

        nnModel = estimator.fit(df)
        res = nnModel.transform(df)

        try:
            tmp_dir = tempfile.mkdtemp()
            modelPath = os.path.join(tmp_dir, "model")
            az_model.save_pytorch(modelPath)
            loaded = TorchNet(modelPath)
            resDF = NNModel(loaded).setPredictionCol("loaded").transform(res)
            assert resDF.filter("prediction==loaded").count() == resDF.count()
        finally:
            try:
                shutil.rmtree(tmp_dir)  # delete directory
            except OSError as exc:
                if exc.errno != errno.ENOENT:  # ENOENT - no such file or directory
                    raise  # re-raise exception

    def test_model_save_load_nnframe(self):
        class SimpleTorchModel(nn.Module):
            def __init__(self):
                super(SimpleTorchModel, self).__init__()
                self.dense1 = nn.Linear(2, 4)
                self.dense2 = nn.Linear(4, 1)

            def forward(self, x):
                x = self.dense1(x)
                x = torch.sigmoid(self.dense2(x))
                return x

        df = self.sqlContext.createDataFrame(
            [(Vectors.dense([2.0, 1.0]), 1.0),
             (Vectors.dense([1.0, 2.0]), 0.0),
             (Vectors.dense([2.0, 1.0]), 1.0),
             (Vectors.dense([1.0, 2.0]), 0.0)],
            ["features", "label"])

        torch_model = SimpleTorchModel()
        torch_criterion = nn.MSELoss()

        az_model = TorchNet.from_pytorch(torch_model, [1, 2])
        az_criterion = TorchCriterion.from_pytorch(torch_criterion, [1, 1], [1, 1])
        estimator = NNEstimator(az_model, az_criterion) \
            .setBatchSize(4) \
            .setLearningRate(0.01) \
            .setMaxEpoch(10)

        nnModel = estimator.fit(df)
        try:
            tmp_dir = tempfile.mkdtemp()
            modelPath = os.path.join(tmp_dir, "model")
            nnModel.save(modelPath)
            loaded = NNModel.load(modelPath)
            resDF = loaded.transform(df)
        finally:
            try:
                shutil.rmtree(tmp_dir)  # delete directory
            except OSError as exc:
                if exc.errno != errno.ENOENT:  # ENOENT - no such file or directory
                    raise  # re-raise exception

    def test_embedding(self):
        model = nn.Embedding(20, 5)
        torch_input = torch.LongTensor(4, 10).random_(0, 20)
        torch_output = model.forward(torch_input).detach().numpy()
        net = TorchNet.from_pytorch(model, [1, 10], dtype=torch.long)
        zoo_output = net.forward(torch_input.numpy())
        assert np.allclose(zoo_output, torch_output)


if __name__ == "__main__":
    pytest.main([__file__])
