import glob
import os
import time
import argparse
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, APIErrorException


if __name__ == '__main__':
    # configure arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--KEY", required=True, help="Access key of endpoint")
    ap.add_argument("-sn", "--SERVICE_NAME", required=True, help="Face service name")
    ap.add_argument("-gid", "--GROUP_ID", required=True, help="Person Group ID to use")
    ap.add_argument("-gn", "--GROUP_NAME", required=True, help="Person Group Name to use")
    ap.add_argument("-f", "--FOLDER", required=True, help="Folder name with faces image")
    args = vars(ap.parse_args())

    # configure the face client
    KEY = args['KEY']
    ENDPOINT = 'https://{0}.cognitiveservices.azure.com/'.format(args['SERVICE_NAME'])
    GROUP_ID = args['GROUP_ID']
    GROUP_NAME = args['GROUP_NAME']
    folder = args['FOLDER']

    face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

    print('Person group:', GROUP_ID)
    try:
        # create a PersonGroup reference class
        face_client.person_group.create(person_group_id=GROUP_ID, name=GROUP_NAME, recognition_model="recognition_03")
    except APIErrorException as err:
        print('person group {0}:{1} already exist.'.format(GROUP_ID, GROUP_NAME))

    # create Person object inside the person group
    fiend = face_client.person_group_person.create(GROUP_ID, folder)

    database = os.path.join(os.getcwd(), 'data', folder)
    path_to_images = glob.glob(os.path.join(database, '*.jpg'))
    for path_ in path_to_images:
        print(path_)
        w = open(path_, 'r+b')
        face_client.person_group_person.add_face_from_stream(GROUP_ID, fiend.person_id, w, detection_model="detection_01")

    print('Training the person group...')
    # Train the person group
    face_client.person_group.train(GROUP_ID)
    while True:
        training_status = face_client.person_group.get_training_status(GROUP_ID)
        print("Training status: {}.".format(training_status.status))
        print()
        if training_status.status is TrainingStatusType.succeeded:
            break
        elif training_status.status is TrainingStatusType.failed:
            print('Training the person group has failed.')
        time.sleep(5)


