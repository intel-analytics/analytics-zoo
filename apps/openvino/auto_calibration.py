#!/usr/bin/env python3
import sys
import os
import argparse
import subprocess
import re
import shutil
import imghdr
import xml.etree.ElementTree as ET


OBJECT_DETECTION = ["fastrcnn",
                    "rfcn",
                    "ssd",
                    "maskrcnn",
                    "yolo"]


def get_calibration_tool_path():
    # Search "/opt/intel/openvino" and home dir
    for path in ["/opt/intel/openvino", os.path.expanduser("~")]:
        res = find_file("calibration_tool", path)
        if res:
            return res
    # current
    raise Exception("ERROR: cannot find calibration_tool from deafult path")


def find_file(name, path):
    """
    Find and return abs path of given name in given path
    """
    if not os.path.exists(path):
        return None
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def get_model_type(model_path):
    """
    Get model type from model path
    """
    regex = re.compile('[^a-zA-Z]')
    alpha_path = regex.sub('', model_path).lower()
    for od_model in OBJECT_DETECTION:
        if od_model in alpha_path:
            return "OD"
    return "C"


def image_classification_val_prepare(image_path):
    """
    Prepare image classification val dir
    find val.txt in dir and return its path
    or current dir path
    """
    if os.path.exists(image_path) and image_path.endswith("txt"):
        return image_path
    for f in os.listdir(image_path):
        curr_path = os.path.join(image_path, f)
        if os.path.isdir(curr_path):
            continue
        if curr_path.endswith("txt"):
            return curr_path
    return image_path


def object_detection_val_prepare(image_path):
    """
    Prepare object detection val dir
    copy *.xml into anno subdir
    copy images (*.png etc) into image subdir
    """
    # Create anno and image dir
    new_image_path = os.path.join(image_path, "images")
    val_image_path = os.path.join(image_path, "images")
    val_anno_path = os.path.join(image_path, "anno")
    val_txt = ""
    folder = ""
    for f in os.listdir(image_path):
        curr_path = os.path.join(image_path, f)
        if os.path.isdir(curr_path):
            continue
        if curr_path.endswith("xml") and len(folder) == 0:
            xml_tree = ET.parse(curr_path)
            doc = xml_tree.getroot()
            folder_element = doc.find("folder")
            if folder_element is not None:
                folder = folder_element.text
    if os.path.exists(val_image_path):
        shutil.rmtree(val_image_path)
    if os.path.exists(val_anno_path):
        shutil.rmtree(val_anno_path)
    if len(folder) != 0:
        val_image_path = os.path.join(val_image_path, folder)
    os.makedirs(val_image_path)
    os.mkdir(val_anno_path)
    for f in os.listdir(image_path):
        curr_path = os.path.join(image_path, f)
        if os.path.isdir(curr_path):
            continue
        # Move *.xml to anno
        if curr_path.endswith("xml"):
            shutil.copy(curr_path, val_anno_path)
        # Move images to image
        elif imghdr.what(curr_path) is not None:
            shutil.copy(curr_path, val_image_path)
        elif curr_path.endswith("txt"):
            val_txt = curr_path
    return (new_image_path, val_anno_path, val_txt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help="Required. OpenVINO IR path, *.xml")
    parser.add_argument('-i', '--input', required=True, help="Required. Path to a directory with \
        validation images. For Classification models, the directory must contain \
            folders named as labels with images inside or a .txt file with a list \
                of images. For Object Detection models, the dataset must be in \
                    VOC format.")
    parser.add_argument('-s', '--subset', help="Number of pictures from the whole \
        validation set tocreate the calibration dataset. Default value is 0, \
            which stands forthe whole provided dataset", default=0)
    parser.add_argument('-o', '--output', help="Output Path for calibrated model")
    parser.add_argument('-t', '--type', help="Type of an inferred network (C by default)\
        C to calibrate Classification network and write the calibrated network to IR\
        OD to calibrate Object Detection network and write the calibrated network to IR\
        RawC to collect only statistics for Classification network and write statistics to IR. With this option, a model is not calibrated.\
        RawOD to collect only statistics for Object Detection network and write statistics to IR.")
    parser.add_argument('--threshold', help="Threshold for a maximum accuracy drop of \
        quantized model. Must be an integer number (percents) without a percent sign. \
            Default value is 1, which stands for accepted accuracy drop in 1%", default=1)

    args = parser.parse_args()

    # Find calibration_tool abs path
    tool_path = get_calibration_tool_path()
    # Check model type (parser from file path)
    model_type = "C"
    if args.type is None:
        model_type = get_model_type(args.model)
    cmd_string = "%s -m %s -t %s" % (tool_path,
                                     args.model, model_type)
    # Handle validation dataset
    if model_type == "OD":
        cmd_string += " -i %s -ODa %s -ODc %s" % object_detection_val_prepare(args.input)
    else:
        cmd_string += " -i %s" % image_classification_val_prepare(args.input)
    # Thredshold
    if args.threshold != 1 and args.threshold > 0:
        cmd_string += " -threshold %d" % args.threshold
    # Subset
    if args.subset != 1 and args.subset > 0:
        cmd_string += " -subset %d" % args.subset
    print(cmd_string)
    # run command
    subprocess.call(cmd_string, shell=True)
