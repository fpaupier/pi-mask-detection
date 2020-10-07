# Pi Mask Detection

This code runs a face detection models as well as a binary classifier mask/no mask on camera
footage from a raspberry Pi to detect whether a person is wearing or not their mask.

For each _event_ - _i.e._, someone not wearing their mask - we store an event in a local SQLite database.

My [alertDispatcher](https://github.com/fpaupier/alertDispatcher) project proposes a service to publish those events to a Kafka topic.

# Installation instruction on the Pi
_Model used: Raspberry Pi 4 model B with 4Go of RAM_

_Python version: 3.7.4_

Install TFlite: follow instructions from [Tensorflow Quickstart](https://www.tensorflow.org/lite/guide/python)

# On my main development machine:
 Install lib for Coral USB accelerator form the [Coral Doc](https://coral.ai/docs/accelerator/get-started)


Retraining an image classification model - here binary classifier mask/no_mask [Coral - retrain classification](https://coral.ai/docs/edgetpu/retrain-classification/#requirements)

# Training process for mak/no mask binary classifier

For dataset of mask/no mask -> Use dataset provide on the [Face Mask Detection project](https://github.com/fpaupier/Face-Mask-Detection)

For the training, I ued Google AutoML Vision, detailed example [here](https://cloud.google.com/vision/automl/docs/edge-quickstart)

# Related projects

This repository hosts the code for the first part of the project; detecting the events at the edge.
It can be completed by the following projects.

- [alertDispatcher](https://github.com/fpaupier/alertDispatcher) a Go module designed to run at the edge, especially a Raspberry Pie 4 B with 4Go of RAM.
The [alertDispatcher](https://github.com/fpaupier/alertDispatcher) polls the local SQLite event store and publishes them to a Kafka topic. 
 
- [alertIngress](https://github.com/fpaupier/alertIngress) an Go module designed to run on a server, consuming from 
a Kafka topic where edge devices pushes their events. Each event consumed by the alert Ingress are archived in PostgresSQL and pushed 
to a notification service.

- [notifyMask](https://github.com/fpaupier/notifyMask) a Go module designed to run on a server, sending email notification to a
system administrator when an event occurs.   