import io

from PIL import Image

import alert_pb2


def main():
    recovered_alert = alert_pb2.Alert()
    with open("serializedAlert", "rb") as fd:
        recovered_alert.ParseFromString(fd.read())

    print(recovered_alert)
    img = Image.open(io.BytesIO(recovered_alert.image.data))
    img.show()


if __name__ == "__main__":
    main()
