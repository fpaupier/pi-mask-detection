import argparse
import datetime
import time

import numpy as np

import detect
from PIL import Image
from PIL import ImageDraw
import tflite_runtime.interpreter as tflite
import platform

from alert import Alert

EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]

def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    return tflite.Interpreter(
        model_path=model_file,
        experimental_delegates=[
            tflite.load_delegate(EDGETPU_SHARED_LIB,
                                 {'device': device[0]} if device else {})
        ])


def draw_objects(draw, objs, labels):
    """Draws the bounding box and label for each object."""
    for obj in objs:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                       outline='red')
        draw.text((bbox.xmin + 10, bbox.ymin + 10),
                  '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                  fill='red')


def get_image(img_path):
    """
    Get the next image to process for the pipeline.

    Fixme: In the first version, load an image from disk. Ultimately, reads from Pi camera feed.
    Returns:

    """
    return Image.open(img_path)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--face_model', required=True,
                        help='File path of face detection .tflite file.')
    parser.add_argument('-m', '--mask_model', required=True,
                        help='File path of mask/no mask binary classifier model .tflite file.')
    parser.add_argument('-d', '--mask_dict', required=True,
                        help='File path of the dict for the binary classifier')
    parser.add_argument('-i', '--input', required=True,
                        help='File path of image to process.')
    parser.add_argument('-t', '--face_threshold', type=float, default=0.4,
                        help='Score threshold for detected objects.')
    parser.add_argument('-o', '--output',
                        help='File path for the result image with annotations')
    parser.add_argument('-c', '--count', type=int, default=5,
                        help='Number of times to run inference')
    parser.add_argument('-l', '--mask_threshold', type=float, default=0.80,
                        help='Threshold value upon which we should raise an alert')
    args = parser.parse_args()
    labels = {}

    # Get camera feed
    image = get_image(img_path=args.input)

    # Apply face detection
    interpreter = make_interpreter(args.face_model)
    interpreter.allocate_tensors()

    scale = detect.set_input(interpreter, image.size,
                             lambda size: image.resize(size, Image.ANTIALIAS))
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    faces = detect.get_output(interpreter, args.face_threshold, scale)
    print('%.2f ms' % (inference_time * 1000))
    print('-------RESULTS--------')
    if not faces:
        print('No Faces detected')
        # TODO/ here add a break statement to come back to acquire a new frame

    for face in faces:
        print('  Face    ')
        print('  score: ', face.score)
        print('  bbox:  ', face.bbox)

    if args.output:
        image = image.convert('RGB')
        draw_objects(ImageDraw.Draw(image), faces, labels)
        image.save(args.output)
        image.show()

    # For each face in the image crop around the ROI and detect if mask or not mask

    # Apply mask / no mask classifier
    mask_interpreter = make_interpreter(args.mask_model)
    mask_interpreter.allocate_tensors()
    input_details = mask_interpreter.get_input_details()
    output_details = mask_interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    for face in faces:
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        region = image.crop(face.bbox).resize((width, height))
        region.show()
        input_data = np.expand_dims(region, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - args.input_mean) / args.input_std

        mask_interpreter.set_tensor(input_details[0]['index'], input_data)

        start_time = time.time()
        mask_interpreter.invoke()
        stop_time = time.time()

        output_data = mask_interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)

        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(args.mask_dict)

        shall_raise_alert = False
        for i in top_k:
            if labels[i] != "no_mask":
                break
            if floating_model:
                res = float(results[i])
            else:
                res = float(results[i] / 255.0)
            if res < args.mask_threshold:
                break
            shall_raise_alert = True
            print('Alert: no mask with probability {:08.6f}: {}'.format(res, labels[i]))

        if not shall_raise_alert:
            pass  # TODO: add a break statement to get to new frame acquisition

        alert = Alert(device="pi",
                      img=region,
                      prob=res,
                      location=(42.3602534, -71.0582912),
                      utc_ts=datetime.datetime.utcnow(),
                      face_detection_model=args.face_model,
                      mask_classifier_model=args.mask_model)
        print(alert)
        print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))

    return


if __name__ == '__main__':
    main()
