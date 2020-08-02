import argparse
import time

import detect
from PIL import Image
from PIL import ImageDraw
import tflite_runtime.interpreter as tflite
import platform

EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]


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
    parser.add_argument('-m', '--model', required=True,
                        help='File path of .tflite file.')
    parser.add_argument('-i', '--input', required=True,
                        help='File path of image to process.')
    parser.add_argument('-t', '--threshold', type=float, default=0.4,
                        help='Score threshold for detected objects.')
    parser.add_argument('-o', '--output',
                        help='File path for the result image with annotations')
    parser.add_argument('-c', '--count', type=int, default=5,
                        help='Number of times to run inference')
    args = parser.parse_args()
    labels = {}

    # Get camera feed
    image = get_image(img_path=args.input)

    # Apply face detection
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()

    scale = detect.set_input(interpreter, image.size,
                             lambda size: image.resize(size, Image.ANTIALIAS))
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    faces = detect.get_output(interpreter, args.threshold, scale)
    print('%.2f ms' % (inference_time * 1000))
    print('-------RESULTS--------')
    if not faces:
        print('No Faces detected')

    for face in faces:
        print('  Face    ')
        print('  score: ', face.score)
        print('  bbox:  ', face.bbox)

    if args.output:
        image = image.convert('RGB')
        draw_objects(ImageDraw.Draw(image), faces, labels)
        image.save(args.output)
        image.show()

    # Apply mask / no mask binary classifier
    pass

    return


if __name__ == '__main__':
    main()
