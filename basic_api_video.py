import cv2
import time
import utils


if __name__ == '__main__':
    KEY = '298b3b2660164139b5b0be02d2c8c219'
    ENDPOINT = 'https://face-api-dev.cognitiveservices.azure.com/'
    vid = cv2.VideoCapture(0)
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
