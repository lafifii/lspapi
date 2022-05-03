import flask 
from flask import request

import mediapipe as mp

app = flask.Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    classes = request.args['classes']

    return 'hi'