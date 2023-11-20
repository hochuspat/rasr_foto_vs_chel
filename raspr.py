from flask import Flask, Response
import cv2
import numpy as np
from sklearn.externals import joblib

app = Flask(__name__)

# Функция для вычисления гистограммы
def calc_hist(img):
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)

# Функция динамической коррекции экспозиции
def adjust_exposure(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    average_brightness = np.mean(gray)
    brightness_threshold = 130

    if average_brightness > brightness_threshold:
        gamma = 0.5
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    else:
        return image

# Загрузка модели обнаружения лиц и классификатора...
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
clf = joblib.load('replay-attack_ycrcb_luv_extraTreesClassifier (1).pkl')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            frame = adjust_exposure(frame)
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            faces3 = net.forward()

            measures = np.zeros(1, dtype=np.float)  # Обнуляем массив перед каждым кадром

            for i in range(faces3.shape[2]):
                confidence = faces3[0, 0, i, 2]
                if confidence > 0.5:
                    box = faces3[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (x, y, x1, y1) = box.astype("int")
                    roi = frame[y:y1, x:x1]

                    img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
                    img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)
            
                    ycrcb_hist = calc_hist(img_ycrcb)
                    luv_hist = calc_hist(img_luv)
            
                    feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
                    feature_vector = feature_vector.reshape(1, len(feature_vector))
            
                    prediction = clf.predict_proba(feature_vector)
                    prob = prediction[0][1]
            
                    measures[0] = prob
            
                    cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)

                    text = "True" if prob < 0.7 else "False"
                    color = (0, 255, 0) if text == "True" else (0, 0, 255)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, text, (x, y-5), font, 0.9, color, 2, cv2.LINE_AA)

            # Вывод в консоль (если требуется)
            print(1 if np.mean(measures) < 0.7 else 0)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Video streaming home page."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Трио представляет</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f0f0f0;
                text-align: center;
                margin: 0;
                padding: 0;
            }
            h1 {
                color: #333;
                margin-top: 20px;
            }
            img {
                border: 4px solid #333;
                border-radius: 8px;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <h1>Video Stream</h1>
        <img src="/video_feed" />
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True)