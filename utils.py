import requests


def get_rectangle(face_info):
    if isinstance(face_info, dict):
        rect = face_info['faceRectangle']
        left = rect['left']
        top = rect['top']
        right = left + rect['width']
        bottom = top + rect['height']
    else:
        rect = face_info.face_rectangle
        left = rect.left
        top = rect.top
        right = left + rect.width
        bottom = top + rect.height
    return (left, top), (right, bottom)


def param_config(face_attributes):
    if face_attributes == '':
        params = {'returnFaceId': 'true', 'returnFaceLandmarks': 'false'}
    else:
        params = {'returnFaceId': 'true', 'returnFaceLandmarks': 'false', 'returnFaceAttributes': face_attributes}
    return params


def detect_face_stream(endpoint, key, image, face_attributes='', recognition_model='recognition_01'):
    face_api_url = endpoint + '/face/v1.0/detect'
    headers = {'Ocp-Apim-Subscription-Key': key, 'Content-Type': 'application/octet-stream'}
    params = param_config(face_attributes)
    params['recognitionModel'] = recognition_model
    response = requests.post(face_api_url, params=params,
                             headers=headers, data=image)
    return response.json()


def detect_face_url(endpoint, key, image_url, face_attributes=''):
    face_api_url = endpoint + '/face/v1.0/detect'
    headers = {'Ocp-Apim-Subscription-Key': key, 'Content-Type': 'application/json'}
    params = param_config(face_attributes)
    response = requests.post(face_api_url, params=params,
                             headers=headers, json={"url": image_url})
    return response.json()


def identify_faces(endpoint, key, group_id, face_id_list):
    face_api_url = endpoint + '/face/v1.0/identify'
    headers = {'Ocp-Apim-Subscription-Key': key, 'Content-Type': 'application/json'}
    body = {
        "personGroupId": group_id,
        "faceIds": face_id_list,
        # "maxNumOfCandidatesReturned": 1,
        # "confidenceThreshold": 0.5
    }
    response = requests.post(face_api_url, headers=headers, json=body)
    return response.json()


def get_person_info(endpoint, key, group_id, candidate_id):
    face_api_url = '{0}/face/v1.0/persongroups/{1}/persons/{2}'.format(endpoint, group_id, candidate_id)
    headers = {'Ocp-Apim-Subscription-Key': key}
    response = requests.get(face_api_url, headers=headers)
    return response.json()
