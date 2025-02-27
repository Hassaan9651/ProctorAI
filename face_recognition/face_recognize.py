# face verification with the VGGFace2 model
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import face_detector as fd
import os
os.chdir("..")


face_model = fd.get_face_detector()
# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
    # detector = MTCNN()
    # detect faces in the image
    # results = detector.detect_faces(pixels)
    results = fd.find_faces(img=pixels, model=face_model)
    # print(results[0])
    # extract the bounding box from the first face
    x1, y1, width, height = results[0] #['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames):
    # extract faces
    faces = [extract_face(f) for f in filenames]
    # convert into an array of samples
    samples = asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    # create a vggface model
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    # perform prediction
    t = time.time()
    yhat = model.predict(samples)
    print("Prediction time:", time.time() - t)
    return yhat


# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
    else:
        print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))


# define filenames
filenames = ["face_recognition/" + x for x in ['h1.jpg', 'h.jpg', "k.jpg"]]
             # 'sharon_stone3.jpg', 'channing_tatum.jpg']
# get embeddings file filenames
import time

t = time.time()
embeddings = get_embeddings(filenames)
print(time.time() - t)
# define sharon stone
sharon_id = embeddings[0]
# verify known photos of sharon
print(embeddings[0].shape)
print('Positive Tests')
t = time.time()
is_match(embeddings[0], embeddings[1])
print("Match Time:", time.time() - t)
# is_match(embeddings[0], embeddings[2])
# verify known photos of other people
print('Negative Tests')
is_match(embeddings[1], embeddings[2])
