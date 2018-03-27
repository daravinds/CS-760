from scipy.io import arff
import sys
import pdb
import math
import random
import numpy

class Logistic():
  learning_rate = 0.0
  epochs = 0
  training_file = None
  test_file = None
  weights = dict()
  weight_differences = dict()
  metadata = None
  features = list()
  bias = 0.0
  mean_xi = dict()
  std_xi = dict()
  feature_type_map = dict()
  negative_label = None
  positive_label = None

  def initialize_variables(self):
    self.learning_rate = 0.0
    self.epochs = 0
    self.training_file = None
    self.test_file = None
    self.weights = dict()
    self.weight_differences = dict()
    self.metadata = None
    self.features = list()
    self.bias = 1.0
    self.mean_xi = dict()
    self.std_xi = dict()
    self.feature_type_map = dict()
    self.negative_label = None
    self.positive_label = None

  def assign_variables(self, learning_rate, epochs, training_file, test_file):
    self.learning_rate = float(learning_rate)
    self.epochs = int(epochs)
    self.training_file = training_file
    self.test_file = test_file

    class_label = self.metadata._attrnames[-1]
    self.negative_label = self.metadata._attributes[class_label][1][0]
    self.positive_label = self.metadata._attributes[class_label][1][1]

    self.features = list(self.metadata._attrnames[:-1])

    feature_details = self.metadata._attributes
    for feature_name in self.features:
      self.feature_type_map[feature_name] = feature_details[feature_name][0]

  def is_real_or_numeric_feature(self, feature_name):
    return self.feature_type_map[feature_name] == "numeric" or self.feature_type_map[feature_name] == "real"

  def is_nominal_feature(self, feature_name):
    return self.feature_type_map[feature_name] == "nominal" or self.feature_type_map[feature_name] == "ordinal"

  def initialize_weights(self):
    # self.weights["bias"] = 0.005
    self.weights["bias"] = random.uniform(-0.01, 0.01)
    for feature in self.features:
      if self.is_real_or_numeric_feature(feature):
        # self.weights[feature] = 0.005
        self.weights[feature] = random.uniform(-0.01, 0.01)
      elif self.is_nominal_feature(feature):
        self.weights[feature] = dict()
        values = self.metadata._attributes[feature][1]
        for value in values:
          self.weights[feature][value] = random.uniform(-0.01, 0.01)

  def normalize(self, data):
    for record in data:
      self.normalize_record(record)

  def normalize_record(self, record):
    for feature in self.features:
      if self.is_real_or_numeric_feature(feature):
        record[feature] = (record[feature] - self.mean_xi[feature]) / self.std_xi[feature]
    return record

  def computeZ(self, record):
    total = self.bias * self.weights["bias"]
    for feature in self.features:
      if self.is_real_or_numeric_feature(feature):
        total += record[feature] * self.weights[feature]
      elif self.is_nominal_feature(feature):
        value = record[feature]
        total += self.weights[feature][value]

    return total      

  def compute_mean_and_standard_deviation(self, data):
    feature_details = self.metadata._attributes

    for feature in self.features:
      if self.is_real_or_numeric_feature(feature):
        total = sum([record[feature] for record in data])
        self.mean_xi[feature] = (total * 1.0) / len(data)
        sum_std = sum([(record[feature] - self.mean_xi[feature])**2 for record in data])
        self.std_xi[feature] = math.sqrt((sum_std * 1.0) / len(data))

  def compute_cross_entropy(self, o, y):
    ce = -y * math.log(o) - (1 - y) * math.log(1 - o)
    return ce

  def perform_logistic_regression(self, learning_rate, epochs, training_file, test_file):
    self.initialize_variables()
    data, self.metadata = read_data(training_file)
    self.assign_variables(learning_rate, epochs, training_file, test_file)
    self.initialize_weights()

    self.compute_mean_and_standard_deviation(data)
    self.normalize(data)
    for loop in range(self.epochs):
      numpy.random.shuffle(data)
      for index, record in enumerate(data):
        o, predicted_label, actual_label = self.predict_instance(record)
        for feature in self.weights.keys():
          weight = self.weights[feature]
          if feature == "bias":
            value = self.bias
          else:
            value = record[feature]

          if isinstance(weight, dict):
            self.weight_differences[feature] = dict()
            feature_values = self.metadata._attributes[feature][1]
            for val in feature_values:
              if val == value:
                self.weight_differences[feature][val] = self.learning_rate * (actual_label - o)
              else:
                self.weight_differences[feature][val] = 0
          else:
            self.weight_differences[feature] = self.learning_rate * (actual_label - o) * value
        for feature in self.weights.keys():
          weight = self.weights[feature]
          if isinstance(weight, dict):
            feature_values = self.metadata._attributes[feature][1]
            for val in feature_values:
              self.weights[feature][val] += self.weight_differences[feature][val]
          else:
            self.weights[feature] += self.weight_differences[feature]

      correct_classifications = int(0)
      total_cross_entropy = 0.0
      for record in data:
        o, predicted_label, actual_label = self.predict_instance(record)
        total_cross_entropy += self.compute_cross_entropy(o, actual_label)
        # print str(o) + " " + str(predicted_label) + " " + str(actual_label)

        if predicted_label == actual_label:
          correct_classifications += 1
      print str(loop + 1) + "\t" + str("%0.9f"%total_cross_entropy) + "\t" + str(correct_classifications) + "\t" + str(len(data) - correct_classifications)


  def predict_instance(self, normalized_record):
    total = self.computeZ(normalized_record)
    o = sigmoid(total)

    if(o <= 0.5):
      predicted_label = 0
    else:
      predicted_label = 1

    if normalized_record[-1] == self.negative_label:
      actual_label = 0
    else:
      actual_label = 1        

    return float(o), int(predicted_label), int(actual_label)


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
    self.normalize(data)
    for record in data:
      o, predicted_label, actual_label = self.predict_instance(record)
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

    precision = (1.0 * tp) / (tp + fp)
    recall = (1.0 * tp) / (tp + fn)
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
  logistic = Logistic()
  learning_rate = sys.argv[1]
  epochs = sys.argv[2]
  training_file = sys.argv[3]
  test_file = sys.argv[4]
  logistic.perform_logistic_regression(learning_rate, epochs, training_file, test_file)
  f1 = logistic.predict()

if __name__ == "__main__":
	classify()
