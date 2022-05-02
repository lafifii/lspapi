import flask 
from flask import request

import cv2


app = flask.Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    classes = request.args['classes']
    version = request.args['version']
    bucket_name = request.args['bucketname']
    object_name = request.args['objectname']
    file_name = request.args['filename']

    return 'hi'