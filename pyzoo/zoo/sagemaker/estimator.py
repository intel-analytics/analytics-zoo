# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import re

from sagemaker.estimator import Framework
# from sagemaker.fw_utils import framework_name_from_image, framework_version_from_tag
from zoo.sagemaker.defaults import ANALYTICS_ZOO_VERSION
from zoo.sagemaker.model import *


class AnalyticsZoo(Framework):
    """Handle end-to-end training and deployment of custom code."""

    __framework_name__ = "analytice-zoo"

    # Hyperparameters
    _num_processes = "sagemaker_num_processes"
    _process_slots_per_host = "sagemaker_process_slots_per_host"
    _additional_mpi_options = "sagemaker_additional_mpi_options"

    def __init__(self, entry_point, source_dir=None, hyperparameters=None, py_version='py2',
                 framework_version=ANALYTICS_ZOO_VERSION, image_name=None, **kwargs):
        """
        This ``Estimator`` executes an Analytics-Zoo script in a managed Analytics execution environment, within a SageMaker
        Training Job. The managed Analytics Zoo environment is an Amazon-built Docker container that executes functions
        defined in the supplied ``entry_point`` Python script.

        Training is started by calling :meth:`~sagemaker.amazon.estimator.Framework.fit` on this Estimator.
        After training is complete, calling :meth:`~sagemaker.amazon.estimator.Framework.deploy` creates a
        hosted SageMaker endpoint and returns an :class:`~zoo.sagemaker..model.AnalyticsZooPredictor` instance
        that can be used to perform inference against the hosted model.

        Technical documentation on preparing Analytics Zoo scripts for SageMaker training and using the Analytics Zoo Estimator is
        available on the project home-page: https://github.com/aws/sagemaker-python-sdk

        Args:
            entry_point (str): Path (absolute or relative) to the Python source file which should be executed
                as the entry point to training. This should be compatible with either Python 2.7 or Python 3.5.
            source_dir (str): Path (absolute or relative) to a directory with any other training
                source code dependencies aside from tne entry point file (default: None). Structure within this
                directory are preserved when training on Amazon SageMaker.
            hyperparameters (dict): Hyperparameters that will be used for training (default: None).
                The hyperparameters are made accessible as a dict[str, str] to the training code on SageMaker.
                For convenience, this accepts other types for keys and values, but ``str()`` will be called
                to convert them before training.
            py_version (str): Python version you want to use for executing your model training code (default: 'py2').
                              One of 'py2' or 'py3'.
            framework_version (str): Analytics zoo version you want to use for executing your model training code.
            image_name (str): If specified, the estimator will use this image for training and hosting, instead of
                selecting the appropriate SageMaker official image based on framework_version and py_version. It can
                be an ECR url or dockerhub image and tag.
                Examples:
                    123.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0
                    custom-image:latest.
            **kwargs: Additional kwargs passed to the :class:`~sagemaker.estimator.Framework` constructor.
        """
        super(AnalyticsZoo, self).__init__(entry_point, source_dir, hyperparameters,
                                      image_name=image_name, **kwargs)
        self.py_version = py_version
        self.framework_version = framework_version

    def hyperparameters(self):
        """Return hyperparameters used by your custom Analytics Zoo code during training."""
        hyperparameters = super(AnalyticsZoo, self).hyperparameters()
        return hyperparameters

    def create_model(self, model_server_workers=None, role=None):
        """Create a SageMaker ``AnalyticsZooModel`` object that can be deployed to an ``Endpoint``.

        Args:
            role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``, which is also used during
                transform jobs. If not specified, the role from the Estimator will be used.
            model_server_workers (int): Optional. The number of worker processes used by the inference server.
                If None, server will use one worker per vCPU.

        Returns:
            zoo.sagemaker.model.AnalyticsZooModel: A SageMaker ``AnalyticsZooModel`` object.
                See :func:`~zoo.sagemaker.model.AnalyticsZooModel` for full details.
        """
        role = role or self.role
        return AnalyticsZooModel(self.model_data, role, self.entry_point, source_dir=self._model_source_dir(),
                            enable_cloudwatch_metrics=self.enable_cloudwatch_metrics, name=self._current_job_name,
                            container_log_level=self.container_log_level, code_location=self.code_location,
                            py_version=self.py_version, framework_version=self.framework_version,
                            model_server_workers=model_server_workers, image=self.image_name,
                            sagemaker_session=self.sagemaker_session)

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details):
        """Convert the job description to init params that can be handled by the class constructor

        Args:
            job_details: the returned job details from a describe_training_job API call.

        Returns:
             dictionary: The transformed init_params

        """
        init_params = super(AnalyticsZoo, cls)._prepare_init_params_from_job_description(job_details)

        image_name = init_params.pop('image')
        framework, py_version, tag = framework_name_from_image(image_name)

        if not framework:
            # If we were unable to parse the framework name from the image it is not one of our
            # officially supported images, in this case just add the image to the init params.
            init_params['image_name'] = image_name
            return init_params

        init_params['py_version'] = py_version
        init_params['framework_version'] = framework_version_from_tag(tag)

        training_job_name = init_params['base_job_name']

        if framework != cls.__framework_name__:
            raise ValueError("Training job: {} didn't use image for requested framework".format(training_job_name))
        return init_params


def framework_version_from_tag(image_tag):
    """Extract the framework version from the image tag.

    Args:
        image_tag (str): Image tag, which should take the form '<framework_version>-<device>-<py_version>'

    Returns:
        str: The framework version.
    """
    tag_pattern = re.compile('^(.*)-(py2|py3)$')
    tag_match = tag_pattern.match(image_tag)
    return None if tag_match is None else tag_match.group(1)


def framework_name_from_image(image_name):
    """Extract the framework and Python version from the image name.

    Args:
        image_name (str): Image URI, which should be one of the following forms:
            legacy:
            '<account>.dkr.ecr.<region>.amazonaws.com/sagemaker-<fw>-<py_ver>-<device>:<container_version>'
            legacy:
            '<account>.dkr.ecr.<region>.amazonaws.com/sagemaker-<fw>-<py_ver>-<device>:<fw_version>-<device>-<py_ver>'
            current:
            '<account>.dkr.ecr.<region>.amazonaws.com/sagemaker-<fw>:<fw_version>-<device>-<py_ver>'

    Returns:
        tuple: A tuple containing:
            str: The framework name
            str: The Python version
            str: The image tag
    """
    # image name format: <account>.dkr.ecr.<region>.amazonaws.com/sagemaker-<framework>-<py_ver>-<device>:<tag>
    sagemaker_pattern = re.compile('^(\d+)(\.)dkr(\.)ecr(\.)(.+)(\.)amazonaws.com(/)(.*:.*)$')
    sagemaker_match = sagemaker_pattern.match(image_name)
    if sagemaker_match is None:
        return None, None, None
    else:
        # extract framework, python version and image tag
        # We must support both the legacy and current image name format.
        name_pattern = re.compile('^sagemaker-analyticszoo:(.*?)-(py2|py3)$')
        legacy_name_pattern = re.compile('^sagemaker-analyticszoo-(py2|py3):(.*)$')
        name_match = name_pattern.match(sagemaker_match.group(8))
        legacy_match = legacy_name_pattern.match(sagemaker_match.group(8))

        if name_match is not None:
            fw, ver,py = name_match.group(1), name_match.group(2), name_match.group(3)
            return fw, py, '{}-{}'.format(ver, py)
        elif legacy_match is not None:
            return legacy_match.group(1), legacy_match.group(2), legacy_match.group(3)
        else:
            return None, None, None