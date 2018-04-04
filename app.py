# app.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import numpy as np
import tensorflow as tf

# # # # # # # 


import logging
import random
import time
import json
import operator

from flask import Flask, jsonify, request, render_template
from werkzeug import secure_filename

# import numpy as np
# from scipy.misc import imread, imresize
# import tensorflow as tf
# from numpy import array


app = Flask(__name__)
app.config.from_object(__name__)

# # This could be added to the Flask configuration
# MODEL_PATH = './retrained_graph.pb'

# # Read the graph definition file
# with open(MODEL_PATH, 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())

# # Load the graph stored in `graph_def` into `graph`
# graph = tf.Graph() 
# with graph.as_default():
#     tf.import_graph_def(graph_def, name='')
    
# # Enforce that no new nodes are added
# graph.finalize()

# # Create the session that we'll use to execute the model
# sess_config = tf.ConfigProto(
#     log_device_placement=False,
#     allow_soft_placement = True,
#     gpu_options = tf.GPUOptions(
#         per_process_gpu_memory_fraction=1
#     )
# )
# sess = tf.Session(graph=graph, config=sess_config)

# # Get the input and output operations
# input_op = graph.get_operation_by_name('input')
# input_tensor = input_op.outputs[0]
# output_op = graph.get_operation_by_name('final_result')
# output_tensor = output_op.outputs[0]

# # All we need to classify an image is:
# # `sess` : we will use this session to run the graph (this is thread safe)
# # `input_tensor` : we will assign the image to this placeholder
# # `output_tensor` : the predictions will be stored here




def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
                input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


@app.route('/classify', methods=['GET', 'POST'])
def classify():

  print('hi made it')

  # app.logger.info("Classifying image %s" % (file_path),)
  # print(file_path)

    # # Load in an image to classify and preprocess it
    # image = imread(file_path)
    # image = imresize(image, [224, 224])
    # image = array(image).reshape(1,150528)
    # image = image.astype(np.float32)
    # image = (image - 128.) / 128.
    # image = image.ravel()
    # images = np.expand_dims(image, 0)
    
    # # Get the predictions (output of the softmax) for this image
    # t = time.time()
    # preds = sess.run(output_tensor, {input_tensor : images})
    # dt = time.time() - t
    # app.logger.info("Execution time: %0.2f" % (dt * 1000.))
    
    # # Single image in this batch
    # predictions = preds[0]
    
    # # The probabilities should sum to 1
    # assert np.isclose(np.sum(predictions), 1)

    # class_label = np.argmax(predictions)
    # app.logger.info("Image %s classified as %d" % (file_path, class_label))
  print(request.files)
  f = request.files['file']
  f.save(secure_filename(f.filename))

  # file_path = request.files['file']

  file_name = f.filename
  model_file = "retrained_graph.pb"
  label_file = "retrained_labels.txt"
  input_height = 224
  input_width = 224
  input_mean = 128
  input_std = 128
  input_layer = "input"
  output_layer = "final_result"

  # parser = argparse.ArgumentParser()
  # parser.add_argument("--image", help="image to be processed")
  # parser.add_argument("--graph", help="graph/model to be executed")
  # parser.add_argument("--labels", help="name of file containing labels")
  # parser.add_argument("--input_height", type=int, help="input height")
  # parser.add_argument("--input_width", type=int, help="input width")
  # parser.add_argument("--input_mean", type=int, help="input mean")
  # parser.add_argument("--input_std", type=int, help="input std")
  # parser.add_argument("--input_layer", help="name of input layer")
  # parser.add_argument("--output_layer", help="name of output layer")
  # args = parser.parse_args()

  # if args.graph:
  #   model_file = args.graph
  # if args.image:
  #   file_name = args.image
  # if args.labels:
  #   label_file = args.labels
  # if args.input_height:
  #   input_height = args.input_height
  # if args.input_width:
  #   input_width = args.input_width
  # if args.input_mean:
  #   input_mean = args.input_mean
  # if args.input_std:
  #   input_std = args.input_std
  # if args.input_layer:
  #   input_layer = args.input_layer
  # if args.output_layer:
  #   output_layer = args.output_layer

  graph = load_graph(model_file)
  t = read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);

  with tf.Session(graph=graph) as sess:
    start = time.time()
    results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
    end=time.time()
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(label_file)

  print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))

  data = {}
  for i in top_k:
    # print(labels[i], results[i])
    data[labels[i].capitalize()] = np.round((results[i]*100), 6)

  data = sorted(data.items(), key=lambda x: x[1], reverse=True)
  print(data)

  return render_template('home.html', data=data)
  # return json.dumps(data)
 

@app.route('/', methods=['GET', 'POST'])
def home():

  return render_template('home.html')




if __name__ == '__main__':

  app.run(debug=True, port=8009)