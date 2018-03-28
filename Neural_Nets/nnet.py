from scipy.io import arff
import sys
import pdb
import math
import random
import numpy

class NeuralNet():
  learning_rate = 0.0
  epochs = 0
  hidden_units = 0
  training_file = None
  test_file = None
  metadata = None
  features = list()
  no_of_features = 0
  bias = 1.0
  mean_xi = dict()
  std_xi = dict()
  feature_type_map = dict()
  negative_label = None
  positive_label = None
  hidden_outputs = []
  weightsXtoH = []
  weightsHtoO = []
  deltaO = None
  deltaH = []

  def initialize_variables(self):
    self.learning_rate = 0.0
    self.epochs = 0
    self.hidden_units = 0
    self.training_file = None
    self.test_file = None
    self.metadata = None
    self.no_of_features = 0
    self.bias = 1.0
    self.mean_xi = dict()
    self.std_xi = dict()
    self.feature_type_map = dict()
    self.negative_label = None
    self.positive_label = None
    self.hidden_outputs = []
    self.weightsXtoH = []
    self.weightsHtoO = []
    self.deltaO = None
    self.deltaH = []

  def assign_variables(self, learning_rate, hidden_units, epochs, training_file, test_file):
    self.learning_rate = float(learning_rate)
    self.hidden_units = int(hidden_units)
    for i in range(self.hidden_units):
      self.hidden_outputs.append(0)
      self.deltaH.append(0)
    self.epochs = int(epochs)
    self.training_file = training_file
    self.test_file = test_file

    class_label = self.metadata._attrnames[-1]
    self.negative_label = self.metadata._attributes[class_label][1][0]
    self.positive_label = self.metadata._attributes[class_label][1][1]

    feature_names = self.metadata._attrnames[:-1]
    feature_details = self.metadata._attributes
    self.no_of_features = 0
    

    for feature_name in feature_names:
      self.feature_type_map[feature_name] = feature_details[feature_name][0]
      if self.is_real_or_numeric_feature(feature_name):
        self.no_of_features += 1
      else:
        self.no_of_features += len(feature_details[feature_name][1])


  def is_real_or_numeric_feature(self, feature_name):
    return self.feature_type_map[feature_name] == "numeric" or self.feature_type_map[feature_name] == "real"

  def is_nominal_feature(self, feature_name):
    return self.feature_type_map[feature_name] == "nominal" or self.feature_type_map[feature_name] == "ordinal"

  def initialize_weights(self):
    for i in range(self.no_of_features + 1):
      self.weightsXtoH.append([])
      for j in range(self.hidden_units):
        self.weightsXtoH[i].append(random.uniform(-0.01, 0.01))

    for j in range(self.hidden_units + 1):
      self.weightsHtoO.append(random.uniform(-0.01, 0.01))

  def compute_cross_entropy(self, o, y):
    ce = -y * math.log(o) - (1 - y) * math.log(1 - o)
    return ce

  def generate_one_of_k_vector(self, all_values, val):
    one_of_k = []
    for value in all_values:
      if value == val:
        one_of_k.append(1)
      else:
        one_of_k.append(0)
    return one_of_k

  def compute_mean_and_standard_deviation(self, data):
    feature_names = self.metadata._attrnames[:-1]
    feature_details = self.metadata._attributes

    for feature in feature_names:
      if self.is_real_or_numeric_feature(feature):
        total = sum([record[feature] for record in data])
        self.mean_xi[feature] = (total * 1.0) / len(data)
        sum_std = sum([(record[feature] - self.mean_xi[feature])**2 for record in data])
        self.std_xi[feature] = math.sqrt((sum_std * 1.0) / len(data))


  def normalize(self, data):
    feature_names = self.metadata._attrnames[:-1]
    feature_details = self.metadata._attributes

    for feature in feature_names:
      if self.is_real_or_numeric_feature(feature):
        for record in data:
          record[feature] = (record[feature] - self.mean_xi[feature]) / self.std_xi[feature]

    normalized_data = []
    for record in data:
      row = []
      row.append(self.bias)
      for feature in feature_names:
        if self.is_real_or_numeric_feature(feature):
          row.append(record[feature])
        else:
          values = feature_details[feature][1]
          row.extend(self.generate_one_of_k_vector(values, record[feature]))
      row.append(record[-1])
      normalized_data.append(row)

    return normalized_data

  def perform_forward_propagation(self, record):
    for j in range(self.hidden_units):
      z = 0.0
      for i in range(self.no_of_features + 1):
        z += (self.weightsXtoH[i][j] * record[i])
      self.hidden_outputs[j] = sigmoid(z)

    zOutput = self.bias * self.weightsHtoO[0]
    for j in range(self.hidden_units):
      zOutput += (self.weightsHtoO[j + 1] * self.hidden_outputs[j])

    o = sigmoid(zOutput)
    if(o <= 0.5):
      predicted_label = 0
    else:
      predicted_label = 1
    if record[-1] == self.negative_label:
      actual_label = 0
    else:
      actual_label = 1        

    return o, int(predicted_label), int(actual_label)

  def compute_errors(self, o, y):
    self.deltaO = float(y - o)
    for j in range(self.hidden_units):
      delta = self.hidden_outputs[j] * (1 - self.hidden_outputs[j]) * self.deltaO * self.weightsHtoO[j + 1]
      self.deltaH[j] = delta

  def compute_gradients(self, record):
    self.weightsHtoO[0] += (self.learning_rate * self.deltaO)
    for j in range(self.hidden_units):
      self.weightsHtoO[j + 1] += (self.learning_rate * self.deltaO * self.hidden_outputs[j])

    for j in range(self.hidden_units):
      for i in range(self.no_of_features + 1):
        self.weightsXtoH[i][j] += (self.learning_rate * self.deltaH[j] * record[i])

  def perform_training(self, learning_rate, hidden_units, epochs, training_file, test_file):
    self.initialize_variables()
    data, self.metadata = read_data(training_file)
    self.assign_variables(learning_rate, hidden_units, epochs, training_file, test_file)
    self.compute_mean_and_standard_deviation(data)
    normalized_data = self.normalize(data)
    self.initialize_weights()

    for loop in range(self.epochs):
      numpy.random.shuffle(normalized_data)

      for index, record in enumerate(normalized_data):
        o, predicted_label, actual_label = self.perform_forward_propagation(record)
        self.compute_errors(o, actual_label)
        self.compute_gradients(record)

      correct_classifications = 0
      total_cross_entropy = 0.0
      for record in normalized_data:
        o, predicted_label, actual_label = self.perform_forward_propagation(record)
        total_cross_entropy += self.compute_cross_entropy(o, actual_label)
        if predicted_label == actual_label:
          correct_classifications += 1
      print str(loop + 1) + "\t" + str("%0.9f"%total_cross_entropy) + "\t" + str(correct_classifications) + "\t" + str(len(data) - correct_classifications)

  def predict(self):
    data, _ = read_data(self.test_file)
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    correct_classifications = int(0)
    normalized_data = self.normalize(data)
    for record in normalized_data:
      o, predicted_label, actual_label = self.perform_forward_propagation(record)
      print str("%0.9f"%o) + "\t" + str(predicted_label) + "\t" + str(actual_label)
      if predicted_label == actual_label:
        correct_classifications += 1
        if predicted_label == 0:
          tn += 1
        else:
          tp += 1
      else:
        if predicted_label == 0:
          fn += 1
        else:
          fp += 1

    if (tp + fp) > 0:
      precision = (1.0 * tp) / (tp + fp)
    if (tp + fn) > 0:
      recall = (1.0 * tp) / (tp + fn)
    if (precision + recall) > 0:
      f1 = float((2.0 * precision * recall) / (precision + recall))
    print str(correct_classifications) + "\t" + str(len(data) - correct_classifications)
    print str(f1)
    return f1

def sigmoid(value):
  return 1.0 / (1.0 + math.exp(-value))

def read_data(file):
  data, metadata = arff.loadarff(open(file, "r"))
  return data, metadata

def classify():
  neural_net = NeuralNet()
  learning_rate = sys.argv[1]
  hidden_units = sys.argv[2]
  epochs = sys.argv[3]
  training_file = sys.argv[4]
  test_file = sys.argv[5]
  neural_net.perform_training(learning_rate, hidden_units, epochs, training_file, test_file)
  f1 = neural_net.predict()

if __name__ == "__main__":
	classify()
