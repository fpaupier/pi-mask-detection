import io
import os
import sys
import time
import yaml
from picamera import PiCamera


import numpy as np

import detect
from PIL import Image
import tflite_runtime.interpreter as tflite
import platform

from alert_encoder import create_alert
from db import LocalDB, persist_alert

ON_DEVICE: bool = True  # Whether we're on the development machine (ON_DEVICE=False), or on the Pi (ON_DEVICE=True)
SLEEP_TIME: int = 1  # second

camera = PiCamera()
camera.rotation = 180  # in degrees, adjust based on your setup

db = LocalDB()

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


def image_to_byte_array(image: Image, fmt: str = "jpeg"):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=fmt)
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


def load_labels(filename: str):
    with open(filename, "r") as f:
        return [line.strip() for line in f.readlines()]


def make_interpreter(model_file: str):
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

    Returns: the image opened

    """
    if ON_DEVICE:
        camera.capture(img_path)
    return Image.open(img_path)


def main():

    face_model = operational_config["models"]["face_detection"]["model"]
    face_threshold = operational_config["models"]["face_detection"]["threshold"]
    mask_model = operational_config["models"]["mask_classifier"]["model"]
    mask_labels = operational_config["models"]["mask_classifier"]["labels"]
    mask_threshold = operational_config["models"]["mask_classifier"]["threshold"]
    deployment: dict = operational_config["deployment"]

    conn = db.conn

    while True:
        # Get camera feed
        image = get_image(img_path="tmp.jpeg")

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
        print("Face detection inference took: %.2f ms" % (inference_time * 1000))
        if not faces:
            print("No Faces detected")
            time.sleep(SLEEP_TIME)

        print("-------RESULTS--------")
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
                    proba = float(results[i])
                else:
                    proba = float(results[i] / 255.0)
                if proba < mask_threshold:
                    break
                shall_raise_alert = True
                print(
                    "Alert: no mask with probability {:08.6f}: {}".format(
                        proba, labels[i]
                    )
                )

            if not shall_raise_alert:
                print("no alerts to raise")
                time.sleep(SLEEP_TIME)
                continue

            alert = create_alert(region, proba)
            persist_alert(conn, alert, deployment["deployed_on"])
            time.sleep(SLEEP_TIME)


if __name__ == "__main__":
    main()
