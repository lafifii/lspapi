from importlib.resources import path
import json
import mediapipe as mp
import cv2
import numpy as np
import json
import requests

from google.cloud import storage
from google.api_core.client_options import ClientOptions
from flask import Flask 
from flask import request

import googleapiclient.discovery

import googleapiclient.discovery

app = Flask(__name__)
GOOGLE_APPLICATION_CREDENTIALS = 'lsp-app-4dbf8-21f5035508f6.json'
BUCKET_NAME = 'lspvideosbucket'

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

def predictVideo(userID, pathIn, pathOut, title,  classes): 
  url = 'https://storage.googleapis.com/' + BUCKET_NAME + '/' + pathIn

  mp_holistic = mp.solutions.holistic 
  
  sentence = []
  sequence = []

  with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    vidcap = cv2.VideoCapture(url)
    success,image = vidcap.read()
    success = True

    cnt = 0

    while success:
      vidcap.set(cv2.CAP_PROP_POS_MSEC,(cnt*30))  
      success, frame = vidcap.read()
      if not success:
        break

      image, results = mediapipe_detection(frame, holistic)
      
      if(results):
        keypoints = get_keypoints(results)
        sequence.append(keypoints)
      
      sequence = sequence[-30:]

      if len(sequence) == 30:
        res = np.expand_dims(sequence, axis=0)
        pred = predict_json('lsp-app-4dbf8', 'us-central1', 'lspmodel', res.tolist())
        sentence.append(classes[np.argmax(pred)])
      
      cnt+=1
  
  storage_client = storage.Client.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)
  bucket = storage_client.bucket(BUCKET_NAME)
  blob = bucket.blob(pathOut)

  info = {
    'userID' : userID,
    'title': title,
    'fileName': pathIn,
    'translation': sentence
  }

  blob.upload_from_string(json.dumps(info))  

@app.route('/', methods=['POST', 'GET'])
def welcome():
  if request.method == 'POST':
    classes = np.array( [ "Alergia", "Baño", "Bien", "Dolor", "Donde", "Gracias", "Hora"])
    content = request.json
    
    userID = content['userID']
    file_name = userID + '/videos/' + content['fileName']
    file_translation = userID + '/translations/' + content['fileName'] 
    title = content['title']
    
    predictVideo(userID, file_name, file_translation, title, classes)
    
    return "Subiendo traduccion  a " + file_translation
  
  else:
    translations = []
    pathUser = request.args['userID'] + '/translations'
    storage_client = storage.Client.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)
    for blob in storage_client.list_blobs(BUCKET_NAME, prefix=pathUser):
      name = blob.name
      url = 'https://storage.googleapis.com/' + BUCKET_NAME + '/' + name
      source = requests.get(url).json()
      translations.append(source)
    
    return json.dumps(translations)
  
if __name__ == '__main__':
  app.run()