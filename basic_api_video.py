import argparse
import cv2
import time
import utils


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--KEY", required=True, help="Access key of endpoint")
    ap.add_argument("-sn", "--SERVICE_NAME", required=True, help="Face service name")
    args = vars(ap.parse_args())

    # configure the face client
    KEY = args['KEY']
    ENDPOINT = 'https://{0}.cognitiveservices.azure.com/'.format(args['SERVICE_NAME'])

    # create a video object and configure size of the output image
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    count = 0
    while True:
        ret, frame = vid.read()
        img = cv2.imencode('.jpg', frame)[1].tobytes()
        detected_faces = utils.detect_face_stream(endpoint=ENDPOINT, key=KEY, image=img)
        print('Image num {} face detected {}'.format(count, detected_faces))
        count += 1
        color = (255, 0, 0)
        thickness = 2
        for face in detected_faces:
            print(face)
            frame = cv2.rectangle(frame, *utils.get_rectangle(face), color, thickness)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        # the 'q' button is set as the quitting button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # wait 3 second to accomplish azure free tier service
        time.sleep(3)

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
