from flask import Flask
from flask import render_template
from flask import Response
import utils
import time

import cv2


app = Flask(__name__)
KEY = '298b3b2660164139b5b0be02d2c8c219'
ENDPOINT = 'https://{0}.cognitiveservices.azure.com/'.format('face-api-dev')
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen():
    global start_time, fps_time, fps
    
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    count = 0
    while True:
        if (time.perf_counter() - fps_time) >= 1 / fps:
            ret, frame = camera.read()
            count += 1
            fps_time = time.perf_counter()
            
            if ret:
                img = cv2.imencode('.jpg', frame)[1].tobytes()
                
                if (time.perf_counter() - start_time) >= 3:
                    detected_faces = utils.detect_face_stream(endpoint=ENDPOINT, key=KEY, image=img)
                    start_time = time.perf_counter()
                
                print('Image num {} face detected {}'.format(count, detected_faces))
                
                color = (255, 0, 0)
                thickness = 2
                for face in detected_faces:
                    print(face)
                    frame = cv2.rectangle(frame, *utils.get_rectangle(face), color, thickness)
                
                img = cv2.imencode('.jpg', frame)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
            else:
                break


if __name__ == "__main__":
    time.sleep(0.1)
    start_time, fps_time, fps = 0, 0, 30
    app.run(debug=True)
