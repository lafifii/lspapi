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
GOOGLE_APPLICATION_CREDENTIALS = 'lsp-app-4dbf8-21f5035508f6.json'

def mediapipe_detection(image, model): # deteccion de mediapipe
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = model.process(image)                 
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

def get_keypoints(results): # concatena todos los keypoints
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])


def predict_json(project, region, model, instances, version=None):
    prefix = "{}-ml".format(region) if region else "ml"
    api_endpoint = "https://{}.googleapis.com".format(prefix)
    client_options = ClientOptions(api_endpoint=api_endpoint)
    service = googleapiclient.discovery.build(
        'ml', 'v1', client_options=client_options)
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']

def predictVideo(pathIn, classes):  
  mp_holistic = mp.solutions.holistic 
  mp_drawing = mp.solutions.drawing_utils 

  sentence = []
  inputs = []
  sequence = []

  threshold = 0.5

  with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True

    fps = vidcap.get(cv2.CAP_PROP_FPS)     
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    cnt = 0

    while success:
      vidcap.set(cv2.CAP_PROP_POS_MSEC,(cnt*30))   
      success, frame = vidcap.read()
      if not success:
        break

      image, results = mediapipe_detection(frame, holistic)
      
      keypoints = get_keypoints(results)
      sequence.append(keypoints)
      
      sequence = sequence[-30:]

      if len(sequence) == 30:
        res = np.expand_dims(sequence, axis=0)
        pred = predict_json('lsp-app-4dbf8', 'us-central1', 'lspmodel', res.tolist())
        sentence.append(classes[np.argmax(pred)])
        
      
      cnt+=1
  
  return sentence

@app.route('/', methods=['GET'])
def home():
    classes = request.args['classes']
    version = request.args['version']
    bucket_name = request.args['bucketname']
    object_name = request.args['objectname']
    file_name = request.args['filename']

    s3 = boto3.client('s3', aws_access_key_id='AKIA4LNJ7XU4LJXKSYEX' , aws_secret_access_key='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')
    s3.download_file(bucket_name, object_name, file_name)
    
    return predictVideo(file_name, classes)