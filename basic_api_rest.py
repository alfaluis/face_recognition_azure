import requests
import json
import cv2
import os

ENDPOINT = 'https://face-api-dev.cognitiveservices.azure.com/'


def get_rectangle(face_dictionary):
    rect = face_dictionary['faceRectangle']
    left = rect['left']
    top = rect['top']
    right = left + rect['width']
    bottom = top + rect['height']
    return (left, top), (right, bottom)


def param_config(face_attributes):
    if face_attributes == '':
        params = {'returnFaceId': 'true', 'returnFaceLandmarks': 'false'}
    else:
        params = {'returnFaceId': 'true', 'returnFaceLandmarks': 'false', 'returnFaceAttributes': face_attributes}
    return params


def detect_face_stream(key, image, face_attributes=''):
    face_api_url = ENDPOINT + 'face/v1.0/detect'
    headers = {'Ocp-Apim-Subscription-Key': key, 'Content-Type': 'application/octet-stream'}
    params = param_config(face_attributes)
    response = requests.post(face_api_url, params=params,
                             headers=headers, data=image)
    return response.json()


def detect_face_url(key, image_url, face_attributes=''):
    face_api_url = ENDPOINT + 'face/v1.0/detect'
    headers = {'Ocp-Apim-Subscription-Key': key, 'Content-Type': 'application/json'}
    params = param_config(face_attributes)
    response = requests.post(face_api_url, params=params,
                             headers=headers, json={"url": image_url})
    return response.json()


if __name__ == '__main__':
    KEY = '298b3b2660164139b5b0be02d2c8c219'
    image_path = os.path.join(os.getcwd(), 'data', 'president-family.jpg')
    # attributes = 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise'
    attributes = ''

    # detect faces in image
    local_image = cv2.imread(image_path)
    img = cv2.imencode('.jpg', local_image)[1].tobytes()
    detected_faces = detect_face_stream(key=KEY, image=img, face_attributes=attributes)

    # Display the resulting frame
    print(detected_faces)
    color = (255, 0, 0)
    thickness = 2
    for face in detected_faces:
        local_image = cv2.rectangle(local_image, *get_rectangle(face), color, thickness)

    cv2.imshow('frame', local_image)
    cv2.waitKey()
    # Destroy all the windows
    cv2.destroyAllWindows()
