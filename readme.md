# Pi Mask Detection

# Installation instruction on the Pi
_Model used: Raspberry Pi 4_

_Python version: 3.7.4_

Install TFlite: follow instructions from [Tensorflow Quickstart](https://www.tensorflow.org/lite/guide/python)

# On my main development machine:
 Install lib for Coral USB accelerator form the [Coral Doc](https://coral.ai/docs/accelerator/get-started)


Retraining an image classification model - here binary classifier mask/no_mask [Coral - retrain classification](https://coral.ai/docs/edgetpu/retrain-classification/#requirements)

# Training process for mak/no mask binary classifier

For dataset of mask/no mask -> Use dataset provide on the [Face Mask Detection project](https://github.com/fpaupier/Face-Mask-Detection)

For the training, I ued Google AutoML Vision, detailed example [here](https://cloud.google.com/vision/automl/docs/edge-quickstart)