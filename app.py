import flask 
from flask import request

import cv2
import mediapipe

app = flask.Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    classes = request.args['classes']

    return 'hi'