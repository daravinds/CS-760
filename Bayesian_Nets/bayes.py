from scipy.io import arff
import sys
import math
import pdb

class NaiveBayes():
  # training_file = None
  # test_file = None
  # metadata = None
  # class_counts = dict()
  # class_probabilities = dict()
  # conditional_counts = dict()
  # conditional_probabilities = dict()
  # curve_data = list()
  # correctly_predicted_count = 0

  def initialize_variables(self):
    self.training_file = None
    self.test_file = None
    self.metadata = None
    self.class_counts = dict()
    self.class_probabilities = dict()
    self.conditional_counts = dict()
    self.conditional_probabilities = dict()
    self.curve_data = list()
    self.correctly_predicted_count = 0

  def perform_naive_bayes(self, data, metadata):
    self.initialize_variables()
    self.metadata = metadata
    class_attribute = self.metadata._attrnames[-1]
    class_values = self.metadata._attributes[class_attribute][1]

    for class_val in class_values:
      conditions = {-1: class_val}
      count = self.get_count(data, conditions)
      self.class_counts[class_val] = float(count)
      self.class_probabilities[class_val] = (self.class_counts[class_val] + 1) / (len(data) + len(class_values))

    features = self.metadata._attrnames[:-1]
    for index, feature in enumerate(features):
      # pdb.set_trace()
      feature_values = self.metadata._attributes[feature][1]
      for class_val in class_values:
        denominator = self.class_counts[class_val] + len(feature_values)
        for feature_val in feature_values:
          conditions = {-1: class_val, index: feature_val}
          count = self.get_count(data, conditions)
          count_with_pseudo = float(count) + 1
          self.initialize_dictionaries(feature, class_val)
          self.conditional_counts[feature][class_val][feature_val] = count_with_pseudo
          self.conditional_probabilities[feature][class_val][feature_val] = count_with_pseudo / denominator

  def initialize_dictionaries(self, feature, class_val):
    if feature not in self.conditional_counts:
      self.conditional_counts[feature] = dict()
      self.conditional_probabilities[feature] = dict()
    if class_val not in self.conditional_counts[feature]:
      self.conditional_counts[feature][class_val] = dict()
      self.conditional_probabilities[feature][class_val] = dict()

  def predict(self, data, metadata):
    class_attribute = metadata._attrnames[-1]
    class_values = metadata._attributes[class_attribute][1]
    positive_label = class_values[0]
    probability_of_classes = dict()
    for feature in self.metadata._attrnames[:-1]:
      print feature + " class"
    print
    self.correctly_predicted_count = 0
    for index, record in enumerate(data):
      actual_label = record[-1]
      max_probability = -1
      predicted_label = None
      denominator = 0
      for class_val in class_values:
        denominator = denominator + self.get_prior_for_record(record, class_val)

      for class_val in class_values:
        numerator = self.get_prior_for_record(record, class_val)
        probability_of_classes[class_val] = float(numerator) / denominator
        if probability_of_classes[class_val] > max_probability:
          max_probability = probability_of_classes[class_val]
          predicted_label = class_val

      probability_of_record_classified_positive = 0
      if predicted_label == positive_label:
        probability_of_record_classified_positive = max_probability
      else:
        probability_of_record_classified_positive = 1 - max_probability
      self.curve_data.append(tuple((actual_label, predicted_label, probability_of_record_classified_positive)))
      if predicted_label == actual_label:
        self.correctly_predicted_count = self.correctly_predicted_count + 1
      print str(predicted_label) + " " + str(actual_label) + " " + str("%0.12f"%max_probability)

    print
    print str(self.correctly_predicted_count)

  def get_prior_for_record(self, record, class_val):
    features = self.metadata._attrnames[:-1]
    product = 1.0
    for feature in features:
      feature_val_in_record = record[feature]
      product = product * self.conditional_probabilities[feature][class_val][feature_val_in_record]
    product = product * self.class_probabilities[class_val]
    return product

  def get_count(self, data, conditions):
    if conditions == 'all':
      return len(data)
    count = 0.0
    for record in data:
      if all([record[index] == value for index, value in conditions.items()]):
        count = count + 1
    return count

class TAN():
  # training_file = None
  # test_file = None
  # metadata = None
  # class_counts = dict()
  # class_probabilities = dict()
  # conditional_counts = dict()
  # conditional_probabilities = dict()
  # mutual_information = dict()
  # root_feature = None
  # parent_mapping = dict()
  # printer = None
  # curve_data = list()
  # correctly_predicted_count = 0

  def initialize_variables(self):
    self.training_file = None
    self.test_file = None
    self.metadata = None
    self.class_counts = dict()
    self.class_probabilities = dict()
    self.conditional_counts = dict()
    self.conditional_probabilities = dict()
    self.mutual_information = dict()
    self.root_feature = None
    self.parent_mapping = dict()
    self.printer = None
    self.curve_data = list()
    self.correctly_predicted_count = 0

  def construct_tree(self):
    visited_vertices = list()
    all_vertices = self.metadata._attrnames[:-1]
    self.root_feature = self.metadata._attrnames[0]
    visited_vertices.append(self.root_feature)
    max_weight = -float('Inf')
    cur_child = None
    cur_parent = None
    while len(visited_vertices) < len(all_vertices):
      max_weight = -float('Inf')
      for vertex in visited_vertices:
        neighbouring_vertices = self.mutual_information[vertex].keys()
        neighbouring_vertices.sort(key=lambda x: (all_vertices.index(x)))
        for child in neighbouring_vertices:
          if child not in visited_vertices:
            mut_info = self.mutual_information[vertex][child]
            if mut_info > max_weight:
              max_weight = mut_info
              cur_parent = vertex
              cur_child = child
      visited_vertices.append(cur_child)
      self.parent_mapping[cur_child] = cur_parent

  def calculate_mutual_information(self, data):
    class_attribute = self.metadata._attrnames[-1]
    class_values = self.metadata._attributes[class_attribute][1]
    for class_val in class_values:
      conditions = {-1: class_val}
      count = self.get_count(data, conditions)
      self.class_counts[class_val] = float(count)
      self.class_probabilities[class_val] = (self.class_counts[class_val] + 1) / (len(data) + len(class_values))

    features = self.metadata._attrnames[:-1]
    for index, feature in enumerate(features):
      feature_values = self.metadata._attributes[feature][1]
      for class_val in class_values:
        denominator = self.class_counts[class_val] + len(feature_values)
        for feature_val in feature_values:
          conditions = {-1: class_val, index: feature_val}
          count = self.get_count(data, conditions)
          count_with_pseudo = float(count) + 1
          self.initialize_dictionaries(index, class_val)
          self.conditional_counts[index][class_val][feature_val] = count_with_pseudo
          self.conditional_probabilities[index][class_val][feature_val] = count_with_pseudo / denominator

    for index1, feature1 in enumerate(features):
      feature_values1 = self.metadata._attributes[feature1][1]
      for index2, feature2 in enumerate(features):
        feature_values2 = self.metadata._attributes[feature2][1]
        for value1 in feature_values1:
          for value2 in feature_values2:
            for class_val in class_values:
              conditions = {-1: class_val, index1: value1, index2: value2}
              left_component = (self.get_count(data, conditions) + 1.0) / (len(data) + len(class_values) * len(feature_values1) * len(feature_values2))
              right_component_numerator = (self.get_count(data, conditions) + 1.0) / (self.class_counts[class_val] + len(feature_values1) * len(feature_values2))
              right_component_denominator = self.conditional_probabilities[index1][class_val][value1] * self.conditional_probabilities[index2][class_val][value2]
              right_component = math.log(float(right_component_numerator) / right_component_denominator, 2)
              self.initialize_mutual_info_dictionary(feature1, feature2)
              if feature1 not in self.mutual_information:
                self.mutual_information[feature1] = dict()
              if feature2 not in self.mutual_information[feature1]:
                self.mutual_information[feature1][feature2] = 0
              self.mutual_information[feature1][feature2] += float(left_component) * right_component

  def initialize_mutual_info_dictionary(self, feature1, feature2):
    if feature1 not in self.mutual_information:
      self.mutual_information[feature1] = dict()
    if feature2 not in self.mutual_information[feature1]:
      self.mutual_information[feature1][feature2] = 0
  
  def initialize_dictionaries(self, index, class_val):
    if index not in self.conditional_counts:
      self.conditional_counts[index] = dict()
      self.conditional_probabilities[index] = dict()
    if class_val not in self.conditional_counts[index]:
      self.conditional_counts[index][class_val] = dict()
      self.conditional_probabilities[index][class_val] = dict()          

  def perform_tan(self, data, metadata):
    self.initialize_variables()
    self.metadata = metadata
    self.calculate_mutual_information(data)
    self.construct_tree()

  def get_prior_for_record(self, data, record, class_val):
    features = self.metadata._attrnames[:-1]
    product = 1.0
    parent_feature = None
    root_feature_index = features.index(self.root_feature)
    root_feature_details = self.metadata._attributes[self.root_feature]
    root_conditions = {-1: class_val, root_feature_index: record[self.root_feature]}
    product = product * (self.get_count(data, root_conditions) + 1) / (self.class_counts[class_val] + len(root_feature_details[1]))
    for index, feature in enumerate(features):
      if feature == self.root_feature:
        continue
      feature_details = self.metadata._attributes[feature]
      feature_val_in_record = record[feature]
      parent_feature = self.parent_mapping[feature]
      parent_feature_details = self.metadata._attributes[parent_feature]
      parent_feature_val_in_record = record[parent_feature]
      parent_feature_index = features.index(parent_feature)
      
      values_for_feature = self.metadata._attributes[feature][1]
      numerator_conditions = {-1: class_val, index: feature_val_in_record, parent_feature_index: parent_feature_val_in_record}
      denominator_conditions = {-1: class_val, parent_feature_index: parent_feature_val_in_record}

      probability_of_feature_given_parent = float(self.get_count(data, numerator_conditions) + 1) / (self.get_count(data, denominator_conditions) + len(values_for_feature))
      product = product * probability_of_feature_given_parent
    product = product * self.class_probabilities[class_val]
    return product

  def predict(self, data, test_data, metadata):
    class_attribute = metadata._attrnames[-1]
    class_values = metadata._attributes[class_attribute][1]
    positive_label = class_values[0]
    probability_of_classes = dict()
    for feature in self.metadata._attrnames[:-1]:
      if feature not in self.parent_mapping:
        print feature + " class"
      else:
        print feature + " " + self.parent_mapping[feature] + " class"
    print
    self.correctly_predicted_count = 0
    for index, record in enumerate(test_data):
      actual_label = record[-1]
      max_probability = -1
      predicted_label = None
      denominator = 0
      for class_val in class_values:
        denominator = denominator + self.get_prior_for_record(data, record, class_val)

      for class_val in class_values:
        numerator = self.get_prior_for_record(data, record, class_val)
        probability_of_classes[class_val] = (1.0 * numerator) / denominator
        if probability_of_classes[class_val] > max_probability:
          max_probability = probability_of_classes[class_val]
          predicted_label = class_val

      probability_of_record_classified_positive = 0
      if predicted_label == positive_label:
        probability_of_record_classified_positive = max_probability
      else:
        probability_of_record_classified_positive = 1 - max_probability
      self.curve_data.append(tuple((actual_label, predicted_label, probability_of_record_classified_positive)))
      if predicted_label == actual_label:
        self.correctly_predicted_count = self.correctly_predicted_count + 1
      print str(predicted_label) + " " + str(actual_label) + " " + str("%0.12f"%max_probability)

    print
    print str(self.correctly_predicted_count)

  def get_count(self, data, conditions):
    if conditions == 'all':
      return len(data)
    count = 0.0
    for record in data:
      if all([record[index] == value for index, value in conditions.items()]):
        count = count + 1
    return count

def read_data(file):
  data, metadata = arff.loadarff(open(file, "r"))
  return data, metadata

# def plot_precision_recall_curve(curve_data, positive_label, bayes):
#   curve_data.sort(key=lambda x: (-x[2]))
#   total_positive_labels = 0
#   for values in curve_data:
#     if values[0] == positive_label:
#       total_positive_labels += 1

#   true_positives_so_far = 0
#   xs = list()
#   ys = list()
#   for index, record in enumerate(curve_data):
#     if record[0] == positive_label:
#       true_positives_so_far += 1
#     precision = float(true_positives_so_far) / (index + 1.0)
#     recall = float(true_positives_so_far) / (total_positive_labels)
#     xs.append(recall)
#     ys.append(precision)

#   if bayes:
#     curve_type = "Naive Bayes"
#   else:
#     curve_type = "Tree Augmented Network"
#   plt.axis([0, 1.2, 0, 1.2])
#   plt.title(str(curve_type) + " Precision-Recall Curve")
#   plt.xlabel("Recall")
#   plt.ylabel("Precision")
#   plt.plot(xs, ys)
#   plt.show()

def classify():
  training_file = sys.argv[1]
  test_file = sys.argv[2]
  is_bayes = sys.argv[3] == 'n'
  if is_bayes:
    naive_bayes = NaiveBayes()
    training_data, training_metadata = read_data(training_file)
    naive_bayes.perform_naive_bayes(training_data, training_metadata)
    test_data, test_metadata = read_data(test_file)
    naive_bayes.predict(test_data, test_metadata)
    class_label = naive_bayes.metadata._attrnames[-1]
    positive_label = naive_bayes.metadata._attributes[class_label][1][0]
    # plot_precision_recall_curve(naive_bayes.curve_data, positive_label, True)
  else:
    tan = TAN()
    training_data, training_metadata = read_data(training_file)
    tan.perform_tan(training_data, training_metadata)
    test_data, test_metadata = read_data(test_file)
    tan.predict(training_data, test_data, test_metadata)
    class_label = tan.metadata._attrnames[-1]
    positive_label = tan.metadata._attributes[class_label][1][0]
    # plot_precision_recall_curve(tan.curve_data, positive_label, False)

def perform_cv():
  file = sys.argv[1]
  data, metadata = read_data(file)
  n = 10
  fold_size = len(data) / n
  start_of_test_data = None
  end_of_test_data = None
  accuracy = dict()
  deltas = list()

  for i in range(n):
    start_of_test_data = i * fold_size
    if i == n - 1:
      end_of_test_data = len(data)
    else:
      end_of_test_data = start_of_test_data + fold_size

    test_data_range = range(start_of_test_data, end_of_test_data)

    test_data = list()
    training_data = list()
    for index, record in enumerate(data):
      if index in test_data_range:
        test_data.append(record)
      else:
        training_data.append(record)

    bayes = NaiveBayes()
    bayes.perform_naive_bayes(training_data, metadata)
    bayes.predict(test_data, metadata)
    accuracy["naive-bayes"] = float(bayes.correctly_predicted_count) / len(test_data)

    tan = TAN()
    tan.perform_tan(training_data, metadata)
    tan.predict(training_data, test_data, metadata)
    accuracy["tan"] = float(tan.correctly_predicted_count) / len(test_data)
    diff = accuracy["naive-bayes"] - accuracy["tan"]
    deltas.append(diff)
    # print "Iteration No:" + str(i+1)
    # print "Accuracy - Naive Bayes: " + str(accuracy["naive-bayes"]) + " TAN: " + str(accuracy["tan"])
    # print "Delta in Naive Bayes accuracy and TAN Accuracy: " + str(diff)

  mean_delta = sum(deltas) / n
  sum_delta_diffs = 0.0
  for delta in deltas:
    sum_delta_diffs += (delta - mean_delta) ** 2
  t_stat = mean_delta / math.sqrt(sum_delta_diffs/(n * (n-1)))
  print mean_delta
  print t_stat

if __name__ == "__main__":
  classify()
  # perform_cv()
