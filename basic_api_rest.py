import os
import argparse
import utils

import cv2


if __name__ == '__main__':
    # configure arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--KEY", required=True, help="Access key of endpoint")
    ap.add_argument("-sn", "--SERVICE_NAME", required=True, help="Face service name")
    args = vars(ap.parse_args())

    # configure the face client
    KEY = args['KEY']
    ENDPOINT = 'https://{0}.cognitiveservices.azure.com/'.format(args['SERVICE_NAME'])

    # load an image
    image_path = os.path.join(os.getcwd(), 'data', 'president-family.jpg')
    local_image = cv2.imread(image_path)
    img = cv2.imencode('.jpg', local_image)[1].tobytes()

    # Function to call the API REST with local image
    attributes = ''
    detected_faces = utils.detect_face_stream(endpoint=ENDPOINT, key=KEY, image=img, face_attributes=attributes)

    # Function to call the API REST with web image
    image_url = "http://www.historyplace.com/kennedy/president-family-portrait-closeup.jpg"
    detected_faces = utils.detect_face_url(endpoint=ENDPOINT, key=KEY, image_url=image_url, face_attributes=attributes)

    # Display the resulting frame
    color = (255, 0, 0)
    thickness = 2
    for face in detected_faces:
        print(face)
        local_image = cv2.rectangle(local_image, *utils.get_rectangle(face), color, thickness)

    cv2.imshow('frame', local_image)
    cv2.waitKey()

    # Destroy all the windows
    cv2.destroyAllWindows()
