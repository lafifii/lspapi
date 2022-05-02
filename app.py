import flask 
from flask import request

import mediapipe as mp
import cv2
import numpy as np

from google.api_core.client_options import ClientOptions
import googleapiclient.discovery

import googleapiclient.discovery

import boto3


app = flask.Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    classes = request.args['classes']
    version = request.args['version']
    bucket_name = request.args['bucketname']
    object_name = request.args['objectname']
    file_name = request.args['filename']

    return 'hi'