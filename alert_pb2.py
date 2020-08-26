# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: alert.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='alert.proto',
  package='maskDetection',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x0b\x61lert.proto\x12\rmaskDetection\"\xd3\x04\n\x05\x41lert\x12\x12\n\nevent_time\x18\x01 \x01(\t\x12/\n\ncreated_by\x18\x02 \x01(\x0b\x32\x1b.maskDetection.Alert.Device\x12/\n\x08location\x18\x03 \x01(\x0b\x32\x1d.maskDetection.Alert.Location\x12\x38\n\x14\x66\x61\x63\x65_detection_model\x18\x04 \x01(\x0b\x32\x1a.maskDetection.Alert.Model\x12\x39\n\x15mask_classifier_model\x18\x05 \x01(\x0b\x32\x1a.maskDetection.Alert.Model\x12\x13\n\x0bprobability\x18\x06 \x01(\x02\x12)\n\x05image\x18\x07 \x01(\x0b\x32\x1a.maskDetection.Alert.Image\x1a\x39\n\x06\x44\x65vice\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\x0c\n\x04guid\x18\x02 \x01(\t\x12\x13\n\x0b\x65nrolled_on\x18\x03 \x01(\t\x1a/\n\x08Location\x12\x11\n\tlongitude\x18\x01 \x01(\x02\x12\x10\n\x08latitude\x18\x02 \x01(\x02\x1a\x36\n\x05Model\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04guid\x18\x02 \x01(\t\x12\x11\n\tthreshold\x18\x03 \x01(\x02\x1a{\n\x05Image\x12\x0e\n\x06\x66ormat\x18\x01 \x01(\t\x12-\n\x04size\x18\x02 \x01(\x0b\x32\x1f.maskDetection.Alert.Image.Size\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\x0c\x1a%\n\x04Size\x12\r\n\x05width\x18\x01 \x01(\x05\x12\x0e\n\x06height\x18\x02 \x01(\x05\x62\x06proto3')
)




_ALERT_DEVICE = _descriptor.Descriptor(
  name='Device',
  full_name='maskDetection.Alert.Device',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='maskDetection.Alert.Device.type', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='guid', full_name='maskDetection.Alert.Device.guid', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='enrolled_on', full_name='maskDetection.Alert.Device.enrolled_on', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=339,
  serialized_end=396,
)

_ALERT_LOCATION = _descriptor.Descriptor(
  name='Location',
  full_name='maskDetection.Alert.Location',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='longitude', full_name='maskDetection.Alert.Location.longitude', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='latitude', full_name='maskDetection.Alert.Location.latitude', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=398,
  serialized_end=445,
)

_ALERT_MODEL = _descriptor.Descriptor(
  name='Model',
  full_name='maskDetection.Alert.Model',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='maskDetection.Alert.Model.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='guid', full_name='maskDetection.Alert.Model.guid', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='threshold', full_name='maskDetection.Alert.Model.threshold', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=447,
  serialized_end=501,
)

_ALERT_IMAGE_SIZE = _descriptor.Descriptor(
  name='Size',
  full_name='maskDetection.Alert.Image.Size',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='width', full_name='maskDetection.Alert.Image.Size.width', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height', full_name='maskDetection.Alert.Image.Size.height', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=589,
  serialized_end=626,
)

_ALERT_IMAGE = _descriptor.Descriptor(
  name='Image',
  full_name='maskDetection.Alert.Image',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='format', full_name='maskDetection.Alert.Image.format', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='size', full_name='maskDetection.Alert.Image.size', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data', full_name='maskDetection.Alert.Image.data', index=2,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_ALERT_IMAGE_SIZE, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=503,
  serialized_end=626,
)

_ALERT = _descriptor.Descriptor(
  name='Alert',
  full_name='maskDetection.Alert',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='event_time', full_name='maskDetection.Alert.event_time', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='created_by', full_name='maskDetection.Alert.created_by', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='location', full_name='maskDetection.Alert.location', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='face_detection_model', full_name='maskDetection.Alert.face_detection_model', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mask_classifier_model', full_name='maskDetection.Alert.mask_classifier_model', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='probability', full_name='maskDetection.Alert.probability', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image', full_name='maskDetection.Alert.image', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_ALERT_DEVICE, _ALERT_LOCATION, _ALERT_MODEL, _ALERT_IMAGE, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=31,
  serialized_end=626,
)

_ALERT_DEVICE.containing_type = _ALERT
_ALERT_LOCATION.containing_type = _ALERT
_ALERT_MODEL.containing_type = _ALERT
_ALERT_IMAGE_SIZE.containing_type = _ALERT_IMAGE
_ALERT_IMAGE.fields_by_name['size'].message_type = _ALERT_IMAGE_SIZE
_ALERT_IMAGE.containing_type = _ALERT
_ALERT.fields_by_name['created_by'].message_type = _ALERT_DEVICE
_ALERT.fields_by_name['location'].message_type = _ALERT_LOCATION
_ALERT.fields_by_name['face_detection_model'].message_type = _ALERT_MODEL
_ALERT.fields_by_name['mask_classifier_model'].message_type = _ALERT_MODEL
_ALERT.fields_by_name['image'].message_type = _ALERT_IMAGE
DESCRIPTOR.message_types_by_name['Alert'] = _ALERT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Alert = _reflection.GeneratedProtocolMessageType('Alert', (_message.Message,), dict(

  Device = _reflection.GeneratedProtocolMessageType('Device', (_message.Message,), dict(
    DESCRIPTOR = _ALERT_DEVICE,
    __module__ = 'alert_pb2'
    # @@protoc_insertion_point(class_scope:maskDetection.Alert.Device)
    ))
  ,

  Location = _reflection.GeneratedProtocolMessageType('Location', (_message.Message,), dict(
    DESCRIPTOR = _ALERT_LOCATION,
    __module__ = 'alert_pb2'
    # @@protoc_insertion_point(class_scope:maskDetection.Alert.Location)
    ))
  ,

  Model = _reflection.GeneratedProtocolMessageType('Model', (_message.Message,), dict(
    DESCRIPTOR = _ALERT_MODEL,
    __module__ = 'alert_pb2'
    # @@protoc_insertion_point(class_scope:maskDetection.Alert.Model)
    ))
  ,

  Image = _reflection.GeneratedProtocolMessageType('Image', (_message.Message,), dict(

    Size = _reflection.GeneratedProtocolMessageType('Size', (_message.Message,), dict(
      DESCRIPTOR = _ALERT_IMAGE_SIZE,
      __module__ = 'alert_pb2'
      # @@protoc_insertion_point(class_scope:maskDetection.Alert.Image.Size)
      ))
    ,
    DESCRIPTOR = _ALERT_IMAGE,
    __module__ = 'alert_pb2'
    # @@protoc_insertion_point(class_scope:maskDetection.Alert.Image)
    ))
  ,
  DESCRIPTOR = _ALERT,
  __module__ = 'alert_pb2'
  # @@protoc_insertion_point(class_scope:maskDetection.Alert)
  ))
_sym_db.RegisterMessage(Alert)
_sym_db.RegisterMessage(Alert.Device)
_sym_db.RegisterMessage(Alert.Location)
_sym_db.RegisterMessage(Alert.Model)
_sym_db.RegisterMessage(Alert.Image)
_sym_db.RegisterMessage(Alert.Image.Size)


# @@protoc_insertion_point(module_scope)
