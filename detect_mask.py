import argparse
import datetime
import io
import os
import sys
import time
import sqlite3
import yaml

import numpy as np

import detect
from PIL import Image
import tflite_runtime.interpreter as tflite
import platform

import alert_pb2

conn = sqlite3.connect("alert.db")

EDGETPU_SHARED_LIB = {
    "Linux": "libedgetpu.so.1",
    "Darwin": "libedgetpu.1.dylib",
    "Windows": "edgetpu.dll",
}[platform.system()]

cur_dir: str = sys.path[0]
CONFIG_PATH: str = os.path.join(cur_dir, "config.yaml")
with open(CONFIG_PATH, "r") as f:
    operational_config = yaml.safe_load(f)
if "device" not in operational_config:
    raise Exception("Failed to load configuration")

DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S.%f"


def image_to_byte_array(image: Image, format: str = "jpeg"):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


def load_labels(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f.readlines()]


def make_interpreter(model_file):
    model_file, *device = model_file.split("@")
    return tflite.Interpreter(
        model_path=model_file,
        experimental_delegates=[
            tflite.load_delegate(
                EDGETPU_SHARED_LIB, {"device": device[0]} if device else {}
            )
        ],
    )


def get_image(img_path):
    """
    Get the next image to process for the pipeline.

    Fixme: In the first version, load an image from disk. Ultimately, reads from Pi camera feed.
    Returns:

    """
    return Image.open(img_path)


def main():

    face_model = operational_config["models"]["face_detection"]["model"]
    face_threshold = operational_config["models"]["face_detection"]["threshold"]
    mask_model = operational_config["models"]["mask_classifier"]["model"]
    mask_labels = operational_config["models"]["mask_classifier"]["labels"]
    mask_threshold = operational_config["models"]["mask_classifier"]["threshold"]
    mask_model_guid: str = operational_config["models"]["mask_classifier"]["guid"]
    face_model_guid: str = operational_config["models"]["face_detection"]["guid"]
    device: dict = operational_config["device"]
    deployment: dict = operational_config["deployment"]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input", required=True, help="File path of image to process."
    )
    parser.add_argument(
        "-o", "--output", help="File path for the result image with annotations"
    )
    args = parser.parse_args()

    # Get camera feed
    image = get_image(img_path=args.input)

    # Apply face detection
    interpreter = make_interpreter(face_model)
    interpreter.allocate_tensors()

    scale = detect.set_input(
        interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS)
    )
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    faces = detect.get_output(interpreter, face_threshold, scale)
    print("%.2f ms" % (inference_time * 1000))
    print("-------RESULTS--------")
    if not faces:
        print("No Faces detected")
        # TODO/ here add a break statement to come back to acquire a new frame

    for face in faces:
        print("  Face    ")
        print("  score: ", face.score)
        print("  bbox:  ", face.bbox)

    image = image.convert("RGB")
    # For each face in the image crop around the ROI and detect if mask or not mask

    # Apply mask / no mask classifier
    mask_interpreter = make_interpreter(mask_model)
    mask_interpreter.allocate_tensors()
    input_details = mask_interpreter.get_input_details()
    output_details = mask_interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]["dtype"] == np.float32

    for face in faces:
        height = input_details[0]["shape"][1]
        width = input_details[0]["shape"][2]
        region = image.crop(face.bbox).resize((width, height))
        input_data = np.expand_dims(region, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - args.input_mean) / args.input_std

        mask_interpreter.set_tensor(input_details[0]["index"], input_data)

        mask_interpreter.invoke()

        output_data = mask_interpreter.get_tensor(output_details[0]["index"])
        results = np.squeeze(output_data)

        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(mask_labels)

        shall_raise_alert = False
        for i in top_k:
            if labels[i] != "no_mask":
                break
            if floating_model:
                res = float(results[i])
            else:
                res = float(results[i] / 255.0)
            if res < mask_threshold:
                break
            shall_raise_alert = True
            print("Alert: no mask with probability {:08.6f}: {}".format(res, labels[i]))

        if not shall_raise_alert:
            pass  # TODO: add a break statement to get to new frame acquisition

        alert = alert_pb2.Alert()

        alert.event_time = datetime.datetime.utcnow().strftime(DATE_FORMAT)

        alert.created_by.type = device["type"]
        alert.created_by.guid = device["guid"]
        alert.created_by.enrolled_on = device["enrolled_on"]

        alert.location.latitude = deployment["latitude"]
        alert.location.longitude = deployment["longitude"]

        alert.face_detection_model.name = face_model
        alert.face_detection_model.guid = face_model_guid
        alert.face_detection_model.threshold = face_threshold

        alert.mask_classifier_model.name = mask_model
        alert.mask_classifier_model.guid = mask_model_guid
        alert.mask_classifier_model.threshold = mask_threshold

        alert.probability = res

        alert.image.format = "jpeg"
        region_width: int = region.size[0]
        region_height: int = region.size[1]
        image_data = image_to_byte_array(region)

        cursor = conn.cursor()
        event_time: str = datetime.datetime.utcnow().strftime(DATE_FORMAT)
        vals = (
            event_time,
            device["type"],
            device["guid"],
            deployment["deployed_on"],
            deployment["longitude"],
            deployment["latitude"],
            face_model,
            face_model_guid,
            face_threshold,
            mask_model,
            mask_model_guid,
            mask_threshold,
            res,
            "jpeg",
            region_width,
            region_height,
            image_data,
        )
        cursor.execute(
            """INSERT INTO 
                alert (created_at, device_type, device_id, device_deployed_on, longitude, latitude, face_model_name, face_model_guid, face_model_threshold, mask_model_name, mask_model_guid, mask_model_threshold, probability, image_format, image_width, image_height, image_data) 
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) """,
            vals,
        )
        conn.commit()

    conn.close()
    return


if __name__ == "__main__":
    main()
