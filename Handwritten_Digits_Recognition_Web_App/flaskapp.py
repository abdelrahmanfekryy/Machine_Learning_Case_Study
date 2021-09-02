from flask import Flask,render_template,request,jsonify,make_response
import cv2
import os
import numpy as np
from keras.models import load_model

print(os.system('set FLASK_APP=flaskapp.py'))

app = Flask(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))
model = load_model(f'{dir_path}/trained_models/LeNet_5.h5')
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])


@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/book", methods=["POST"])
def create_entry():
    if request.method == "POST":
        req = request.get_json()
        image = np.array(list(req.values())).reshape(400,400,4).astype('uint8')
        image = (cv2.resize(image,dsize=(32,32), interpolation=cv2.INTER_LINEAR)/255).max(axis=-1)
        y_pred = np.argmax(model.predict(np.expand_dims(image.reshape(32,32,1),axis=0)), axis=-1)
        res = make_response(str(y_pred[0]), 200)
        print('jsonify',res)
        return res

if __name__ == "__main__":
    app.run(debug=True, port=5000)