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

import re
import os
import numpy as np

class MKLSetting(object):
    def __init__(self, cores):
        self.cores = cores

    def get_cores(self):
        if self.cores:
            return self.cores
        if "OMP_NUM_THREADS" in os.environ :
            return int(os.environ.get("OMP_NUM_THREADS"))
        return 0

def to_list(input):
    if isinstance(input, (list, tuple)):
        return list(input)
    else:
        return [input]

def unflatten(vector, shapes):
    """
    :param vector: 1D tensor
    :param shapes:
    :return: list of ndarray
    """
    i = 0
    arrays = []
    for shape in shapes:
        size = np.prod(shape, dtype=np.int)
        array = vector[i:(i + size)].reshape(shape)
        arrays.append(array)
        i += size
    return arrays


def flatten(inputs):
    """
    :param inputs: list of ndarray
    :return:
    """
    return np.concatenate([
        i.flatten() for i in inputs])

def split(v, num_slices):
    """
    Split v as evently as possible
    :param v: An 1-D vector
    :return: list of vectors
    """
    # np.split(grads, self.num_worker would raise exception if grads cannot be evenly divided.
    #
    vshape = v.shape
    assert len(vshape) == 1, "we only accept 1D vector here"
    vector_len = vshape[0]
    slice_len = vector_len // num_slices
    slice_extra_len = vector_len % num_slices
    # i.e vector_len = 10, num_slices=20, then we should return [10]
    if slice_len == 0:
        return [v]
    else:
        result_slices = []
        for i in range(0, num_slices):
            len_tmp = slice_len + (1 if i < slice_extra_len else 0)
            offset = i * slice_len + min(i, slice_extra_len)
            result_slices.append(v[offset:(offset + len_tmp)])
        return result_slices


def resourceToBytes(resource_str):
    matched = re.compile("([0-9]+)([a-z]+)?").match(resource_str.lower())
    fraction_matched = re.compile("([0-9]+\\.[0-9]+)([a-z]+)?").match(resource_str.lower())
    if fraction_matched:
        raise Exception(
            "Fractional values are not supported. Input was: {}".format(resource_str))
    try:
        value = int(matched.group(1))
        postfix = matched.group(2)
        if postfix == 'b':
            value = value
        elif postfix == 'k':
            value = value * 1000
        elif postfix == "m":
            value = value * 1000 * 1000
        elif postfix == 'g':
            value = value * 1000 * 1000 * 1000
        else:
            raise Exception("Not supported type: {}".format(resource_str))
        return value
    except Exception:
        raise Exception("Size must be specified as bytes(b),"
                        "kilobytes(k), megabytes(m), gigabytes(g). "
                        "E.g. 50b, 100k, 250m, 30g")
