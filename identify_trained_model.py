import os
import utils
import cv2
import argparse


if __name__ == '__main__':
    # configure arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--KEY", required=True, help="Access key of endpoint")
    ap.add_argument("-sn", "--SERVICE_NAME", required=True, help="Face service name")
    ap.add_argument("-gid", "--GROUP_ID", required=True, help="Person Group ID to use")
    ap.add_argument("-im", "--IMG_NAME", required=True, help="name of image to test")
    args = vars(ap.parse_args())

    # configure the face client
    KEY = args['KEY']
    ENDPOINT = 'https://{0}.cognitiveservices.azure.com/'.format(args['SERVICE_NAME'])
    GROUP_ID = args['GROUP_ID']
    img_name = args['IMG_NAME']

    # path to the image
    database = os.path.join(os.getcwd(), 'data')
    image_path = os.path.join(database, img_name)
    # Load image
    local_image = cv2.imread(image_path)
    img = cv2.imencode('.jpg', local_image)[1].tobytes()

    # Function to call the API REST with local image
    attributes = ''
    detected_faces = utils.detect_face_stream(endpoint=ENDPOINT, key=KEY, image=img, face_attributes=attributes,
                                              recognition_model='recognition_03')

    # Identify faces from the detected
    faces_ids = [f['faceId'] for f in detected_faces]
    identify_output = utils.identify_faces(endpoint=ENDPOINT, key=KEY, group_id=GROUP_ID, face_id_list=faces_ids)
    print(identify_output)

    thickness = 2
    for person in identify_output:
        print('Result of face: {0}.'.format(person['faceId']))
        face = [face for face in detected_faces if face['faceId'] == person['faceId']][0]
        if not len(person['candidates']) == 0:
            candidate = person['candidates'][0]
            print('Identified in {} with a confidence: {}.'.format(person['faceId'], candidate['confidence']))
            person_info = utils.get_person_info(ENDPOINT, KEY, GROUP_ID, candidate['personId'])
            print('Name Group person identified: {0}'.format(person_info['name']))
            color = (0, 255, 0)
            local_image = cv2.rectangle(local_image, *utils.get_rectangle(face), color, thickness)
            x, y = face['faceRectangle']['left'], face['faceRectangle']['top'] - 5
            cv2.putText(local_image, person_info['name'], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        else:
            print('no match found')
            color = (255, 0, 0)
            local_image = cv2.rectangle(local_image, *utils.get_rectangle(face), color, thickness)

    cv2.imshow('frame', local_image)
    cv2.waitKey()
