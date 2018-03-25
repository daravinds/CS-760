from scipy.io import arff
import sys
import pdb
import math
import random

class NeuralNet():
  learning_rate = 0.0
  epochs = 0
  training_file = None
  test_file = None
  weights = dict()
  weight_differences = dict()
  # no_of_features = 0
  # no_of_values = dict()
  metadata = None
  features = list()
  bias = 0.0
  mean_xi = dict()
  std_xi = dict()
  feature_type_map = dict()

  def initialize_variables(self):
    self.learning_rate = 0.0
    self.epochs = 0
    self.training_file = None
    self.test_file = None
    self.weights = dict()
    # self.no_of_features = 0
    # self.no_of_values = dict()
    self.metadata = None
    self.features = list()
    self.bias = 0.0
    self.mean_xi = dict()
    self.std_xi = dict()
    self.feature_type_map = dict()

  def assign_variables(self, learning_rate, epochs, training_file, test_file):
    self.learning_rate = float(learning_rate)
    self.epochs = int(epochs)
    self.training_file = training_file
    self.test_file = test_file
    self.features = self.metadata._attrnames[:-1]

    feature_details = self.metadata._attributes
    for feature_name in self.features:
      self.feature_type_map[feature_name] = feature_details[feature_name][0]
      # feature_values = feature_details[feature_name][1]

      # if self.is_real_or_numeric_feature(feature_name):
      #   self.no_of_values[feature_name] = 1
      # elif self.is_nominal_feature(feature_name):
      #   self.no_of_values[feature_name] = len(feature_values)

      # self.no_of_features += self.no_of_values[feature_name]

  def is_real_or_numeric_feature(self, feature_name):
    return self.feature_type_map[feature_name] == "numeric" or self.feature_type_map[feature_name] == "real"

  def is_nominal_feature(self, feature_name):
    return self.feature_type_map[feature_name] == "nominal" or self.feature_type_map[feature_name] == "ordinal"

  def initialize_weights_and_bias(self):
    self.bias = 1.0
    self.weights["bias"] = -0.003 # random.uniform(-0.01, 0.01)
    for feature in self.features:
      if self.is_real_or_numeric_feature(feature):
        self.weights[feature] = -0.003 # random.uniform(-0.01, 0.01)
      elif self.is_nominal_feature(feature):
        self.weights[feature] = dict()
        values = self.metadata._attributes[feature][1]
        for value in values:
          self.weights[feature][value] = random.uniform(-0.01, 0.01)

  def normalize_record(self, record):
    for feature in self.features:
      if feature == "bias":
        continue
      if self.is_real_or_numeric_feature(feature):
        record[feature] = (record[feature] - self.mean_xi[feature]) / self.std_xi[feature]
    return record

  def computeZ(self, record):
    # record = self.normalize_record(record)
    total = self.bias * self.weights["bias"]
    for feature in self.features:
      if self.is_real_or_numeric_feature(feature):
        # value = (record[feature] - self.mean_xi[feature]) / self.std_xi[feature]
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
      # elif self.is_nominal_feature(feature):
      #   self.mean_xi[feature] = dict()
      #   self.std_xi[feature] = dict()
      #   values_count = dict()
      #   for record in data:
      #     values_count[record[feature]] = values_count.get(record[feature], 0) + 1
      #   feature_values = feature_details[feature][1]
      #   for value in feature_values:
      #     value_count = values_count.get(value, 0)
      #     self.mean_xi[feature][value] = (value_count * 1.0) / len(data)
      #     self.std_xi[feature][value] = math.sqrt((value_count * (1 - self.mean_xi[feature][value])**2 * 1.0) / len(data))


  def compute_cross_entropy(self, o, y):
    if o == 1 or o == 0:
      return 0

    ce = y * math.log(o) + (1 - y) * math.log(1 - o)
    return -ce

  def perform_logistic_regression(self, learning_rate, epochs, training_file, test_file):
    self.initialize_variables()
    data, self.metadata = read_data(training_file)
    self.assign_variables(learning_rate, epochs, training_file, test_file)
    self.initialize_weights_and_bias()
    # random.shuffle(data)
    class_label = self.metadata._attrnames[-1]
    class_labels = self.metadata._attributes[class_label][1]

    self.compute_mean_and_standard_deviation(data)
    for loop in range(self.epochs):
      # random.shuffle(data)
      correct_classifications = 0
      total_cross_entropy = 0.0
      for record in data[:1]:
        pdb.set_trace()
        record = self.normalize_record(record)
        # pdb.set_trace()
        y = record[-1]
        yInt = class_labels.index(y)
        z = self.computeZ(record)
        o = sigmoid(z)
        if(o <= 0.5):
          label = 0
        else:
          label = 1
        if label == yInt:
          # print "o:" + str(o) + " y:" + str(y)
          correct_classifications += 1

        # pdb.set_trace()
        error = self.compute_cross_entropy(o, yInt)
        total_cross_entropy += error

        for feature in self.weights.keys():
          weight = self.weights[feature]
          if feature == "bias":
            value = self.bias
          else:
            # value = (record[feature] - self.mean_xi[feature]) / self.std_xi[feature]
            value = record[feature]

          if isinstance(weight, dict):
            self.weight_differences[feature] = dict()
            feature_values = self.metadata._attributes[feature][1]
            for val in feature_values:
              if val == value:
                self.weight_differences[feature][val] = self.learning_rate * (yInt - o)
              else:
                self.weight_differences[feature][val] = 0
          else:
            self.weight_differences[feature] = self.learning_rate * (yInt - o) * value

        for feature in self.weights.keys():
          weight = self.weights[feature]
          if isinstance(weight, dict):
            feature_values = self.metadata._attributes[feature][1]
            for val in feature_values:
              self.weights[feature][val] += self.weight_differences[feature][val]
          else:
            self.weights[feature] += self.weight_differences[feature]
      print str(loop + 1) + "\t" + str(total_cross_entropy) + "\t" + str(correct_classifications) + "\t" + str(len(data) - correct_classifications)

  def predict(self):
    data, _ = read_data(self.test_file)
    class_label = self.metadata._attrnames[-1]
    class_labels = self.metadata._attributes[class_label][1]
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    total_positives = 0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    for record in data:
      record = self.normalize_record(record)
      total = 0.0
      for feature in self.weights.keys():
        if feature == "bias":
          value = self.bias
        else:
          value = record[feature]
        total += value * self.weights[feature]
      # pdb.set_trace()
      o = sigmoid(total)
      if(o <= 0.5):
        label = 0
      else:
        label = 1
      predicted_class = class_labels[label]
      actual_class = record[-1]
      if actual_class == class_labels[1]:
        total_positives += 1
      print str(o) + " " + predicted_class + " " + actual_class

      if predicted_class == actual_class:
        if predicted_class == class_labels[1]:
          tp += 1
        else:
          tn += 1
      else:
        if predicted_class == class_labels[1]:
          fp += 1
        else:
          fn += 1
    precision = (1.0 * tp) / (tp + fp)
    recall = (1.0 * tp) / (total_positives)
    f1 = 2 * precision * recall / (precision + recall)
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
  epochs = sys.argv[2]
  training_file = sys.argv[3]
  test_file = sys.argv[4]
  avg_f1 = 0.0
  f1 = 0.0

  neural_net.perform_logistic_regression(learning_rate, epochs, training_file, test_file)
  f1 = neural_net.predict()

if __name__ == "__main__":
	classify()
