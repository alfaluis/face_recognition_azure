import os
import requests
import argparse
from io import BytesIO
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import utils


if __name__ == '__main__':
    # configure arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--KEY", required=True, help="Access key of endpoint")
    ap.add_argument("-ep", "--ENDPOINT", required=True, help="Endpoint to the face service")
    args = vars(ap.parse_args())

    # configure the face client
    KEY = args['KEY']
    ENDPOINT = args['ENDPOINT']
    face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

    # make prediction from URL
    multi_face_image_url = "http://www.historyplace.com/kennedy/president-family-portrait-closeup.jpg"
    multi_image_name = os.path.basename(multi_face_image_url)
    detected_faces = face_client.face.detect_with_url(url=multi_face_image_url)

    # make prediction from URL
    local_image_path = os.path.join(os.getcwd(), 'data', 'president-family.jpg')
    image = open(local_image_path, 'r+b')
    detected_faces = face_client.face.detect_with_stream(image)

    # download the image from web source
    response = requests.get(multi_face_image_url)
    img = Image.open(BytesIO(response.content))

    # For each face returned use the face rectangle and draw a red box.
    draw = ImageDraw.Draw(img)
    for face in detected_faces:
        print(face)
        draw.rectangle(utils.get_rectangle(face), outline='red')

    # Display the image in the users default image browser.
    img.show()


