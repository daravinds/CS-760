from scipy.io import arff
# import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import random

CLASS_INDEX = -1
class Node():
  feature = None
  description = None # <= or > or =
  val = None
  pos_count = 0 # number of +ve instances at this node
  neg_count = 0 # number of -ve instances at this node
  leaf = False # is leaf node
  label = None # Class label for leaf node
  children = None # children of a node

class DecisionTree():
  root = None
  metadata = None
  training_file = None
  test_file = None
  min_instances = 0
  correct_predictions = 0
  wrong_predictions = 0
  dataset_size = 0

  def learn(self, training_file, min_instances, print_text, fraction = None):
    self.training_file = training_file
    self.min_instances = int(min_instances)
    data, self.metadata = arff.loadarff(open(self.training_file, "r"))
    if(fraction != None):
      data = self.pick_random_fraction(data, fraction)
    self.dataset_size = len(data)
    self.all_attributes = self.metadata._attrnames
    self.root = self.build_tree(data, self.all_attributes[:-1], '+')
    if(print_text):
      self.print_node(0, self.root)

  def pick_random_fraction(self, data, fraction):
    taken = {}
    total_size = len(data)
    expected_size = fraction * total_size
    list = []
    while(len(list) < expected_size):
      random_index = random.randint(0, total_size - 1)
      if(not taken.get(random_index), False):
        list.append(data[random_index])
        taken[random_index] = True
    return list

  def predict(self, test_file, print_text):
    test_label_confidence_pairs = []
    self.test_file = test_file
    data, self.metadata = arff.loadarff(open(self.test_file, "r"))
    correctly_classified_count = 0
    misclassified_count = 0
    if(print_text):
      print("<Predictions for the Test Set Instances>")
    for index, record in enumerate(data):
      target_leaf = self.get_label(self.root, record)
      predicted_label = target_leaf.label
      actual_label = record[CLASS_INDEX]
      confidence = (target_leaf.pos_count + 1.0) / (target_leaf.pos_count + target_leaf.neg_count + 2.0)
      test_label_confidence_pairs.append((actual_label, confidence))
      if(print_text):
        print(str(index + 1) + ": Actual: " + str(actual_label) + " Predicted: " + str(predicted_label))
      if predicted_label == actual_label:
        correctly_classified_count = correctly_classified_count + 1
      else:
        misclassified_count = misclassified_count + 1
    self.correct_predictions = correctly_classified_count
    self.wrong_predictions = misclassified_count
    if(print_text):
      sys.stdout.write("Number of correctly classified: " + str(correctly_classified_count) + " Total number of test instances: " + str(len(data)))
    return test_label_confidence_pairs

  def get_label(self, node, record):
    if(node.leaf):
      return node
    else:
      value = record[node.feature]
      next_node = None
      if(" = " in node.children[0].description):
        for child in node.children:
          if child.val == value:
            next_node = child
      else:
        if value <= node.children[0].val:
          next_node = node.children[0]
        else:
          next_node = node.children[1]
    return self.get_label(next_node, record)

  def print_node(self, depth, node):
    if(node.children == None):
      return
    for new_node in node.children:
      text = depth * '|\t'
      text += str(node.feature.lower())
      text += str(new_node.description)
      if("<" in new_node.description or ">" in new_node.description):
        text += str("%0.6f"%new_node.val)
      else:
        text += str(new_node.val)
      text += " [" + str(new_node.pos_count) + " " + str(new_node.neg_count) + "]"
      if(new_node.leaf):
        text += ": " + str(new_node.label)
      print(text)
      self.print_node(depth + 1, new_node)

  def get_positive_and_negative_counts(self, data):
    pos_count = 0
    neg_count = 0
    for record in data:
      if(record[CLASS_INDEX] == '+'):
        pos_count = pos_count + 1
      elif(record[CLASS_INDEX] == '-'):
        neg_count = neg_count + 1
    return pos_count, neg_count

  # def print_node_details(self, node):
  #   print("feature:"  + str(node.feature))
  #   print("description:"  + str(node.description))
  #   print("val:"  + str(node.val))
  #   print("pos_count:"  + str(node.pos_count))
  #   print("neg_count:"  + str(node.neg_count))
  #   print("leaf:"  + str(node.leaf))
  #   print("label:"  + str(node.label))
  #   print("children count:"  + str(len(node.children)))

  def build_tree(self, data, attributes, parent_majority_class):
    node = Node()
    node.pos_count, node.neg_count = self.get_positive_and_negative_counts(data)
    target_vals = [record[CLASS_INDEX] for record in data]
    major_class = self.major_class(data, parent_majority_class, target_vals)
    if(len(data) < self.min_instances or len(attributes) == 0):
      node.leaf = True
      node.label = major_class
      return node
    elif(len(np.unique(target_vals)) == 1):
      node.leaf = True
      node.label = major_class
      return node
    else:
      best, threshold = self.choose_best_attribute(data, attributes)
      index_of_best = self.all_attributes.index(best)
      node.feature = best
      if self.is_nominal_attribute(index_of_best):
        new_attributes = [attribute for attribute in attributes if attribute != best]
      else:
        new_attributes = attributes
      node.children = []
      if(self.is_nominal_attribute(index_of_best)):
        for val in self.get_values(data, best):
          new_data = self.get_data(data, best, val)
          child = self.build_tree(new_data, new_attributes, major_class)
          child.val = val
          child.description = " = "
          node.children.append(child)
      elif(self.is_numeric_attribute(index_of_best)):
        left_data = [record for record in data if record[index_of_best] <= threshold]
        left_child = self.build_tree(left_data, new_attributes, major_class)
        left_child.val = threshold
        left_child.description = " <= "
        node.children.append(left_child)
        right_data = [record for record in data if record[index_of_best] > threshold]
        right_child = self.build_tree(right_data, new_attributes, major_class)
        right_child.val = threshold
        right_child.description = " > "
        node.children.append(right_child)
      return node

  def get_data(self, data, best, val):
    results = []
    attribute_index = self.all_attributes.index(best)
    for record in data:
      if(record[attribute_index] == val):
        results.append(record)
    return results

  def get_values(self, data, best):
    return self.metadata._attributes[best][1]
    # values_in_order = self.metadata._attributes[best][1]
    # frequency_map = self.get_frequency_by_attribute(data, self.all_attributes.index(best))
    # values_present = frequency_map.keys()
    # values = [value for value in values_in_order if value in values_present]
    # return values

  def choose_best_attribute(self, data, attributes):
    max_gain = -1
    attribute_with_max_gain = None
    threshold = None
    for attribute in attributes:
      gain, thresh = self.info_gain(data, attribute)
      if(gain > max_gain):
        max_gain = gain
        attribute_with_max_gain = attribute
        threshold = thresh

    return attribute_with_max_gain, threshold

  def info_gain(self, data, attribute):
    if(self.is_nominal_attribute(self.all_attributes.index(attribute))):
      gain = self.info_gain_for_nominal_attribute(data, attribute)
      threshold = None
    elif(self.is_numeric_attribute(self.all_attributes.index(attribute))):
      gain, threshold = self.info_gain_for_numeric_attribute(data, attribute)
    return gain, threshold

  def info_gain_for_numeric_attribute(self, data, attribute):
    new_data = np.sort(data, order = attribute)
    threshold = None
    max_gain = -1
    entropy = self.entropy(new_data)
    for index in range(1, len(new_data)):
      # if(new_data[index - 1][CLASS_INDEX] != new_data[index][CLASS_INDEX]):
      thresh = (new_data[index - 1][attribute] + new_data[index][attribute]) / 2.0
      objects_lte_threshold, objects_gt_threshold = self.get_objects(new_data, thresh, attribute)
      posterior_entropy = float(len(objects_lte_threshold)) / len(new_data) * self.entropy(objects_lte_threshold) + float(len(objects_gt_threshold)) / len(new_data) * self.entropy(objects_gt_threshold)
      gain = entropy - posterior_entropy
      if(gain > max_gain):
        max_gain = gain
        threshold = thresh
    return max_gain, threshold

  def get_objects(self, data, threshold, attribute):
    lte_objects = [record for record in data if record[attribute] <= threshold]
    gt_objects = [record for record in data if record[attribute] > threshold]
    return lte_objects, gt_objects

  def info_gain_for_nominal_attribute(self, data, attribute):
    posterior_entropy = 0.0
    attribute_index = self.all_attributes.index(attribute)
    frequency_map = self.get_frequency_by_attribute(data, attribute_index)
    values_in_order = self.metadata._attributes[attribute][1]
    values_present = frequency_map.keys()
    values = [value for value in values_in_order if value in values_present]
    for attribute_value in values:
      fraction_of_type = float(frequency_map.get(attribute_value)) / len(data)
      objects_of_type = [record for record in data if record[attribute_index] == attribute_value]
      posterior_entropy += fraction_of_type * self.entropy(objects_of_type)
    return self.entropy(data) - posterior_entropy

  def entropy(self, data):
    frequency_map = self.get_class_frequencies(data)
    entropy_of_data = 0.0
    for frequency in frequency_map.values():
      fraction_of_type = float(frequency) / len(data)
      entropy_of_data += (fraction_of_type * math.log(fraction_of_type, 2))
    entropy_of_data = - entropy_of_data
    return entropy_of_data

  def major_class(self, data, parent_majority_class, target_vals):
    if(len(data) == 0):
      return parent_majority_class
    pos_count = target_vals.count('+')
    neg_count = target_vals.count('-')
    if(pos_count > neg_count):
      return '+'
    elif(pos_count < neg_count):
      return '-'
    else:
      return parent_majority_class
    # class_frequencies = self.get_class_frequencies(data)
    # if(len(np.unique(class_frequencies.values()) <= 1)):
    #   return parent_majority_class
    # class_with_max_frequency = None
    # max_frequency = 0

    # for class_type in class_frequencies.keys():
    #   if class_frequencies.get(class_type) > max_frequency:
    #     max_frequency = class_frequencies.get(class_type)
    #     class_with_max_frequency = class_type
    # return class_with_max_frequency

  def get_class_frequencies(self, data):
    return self.get_frequency_by_attribute(data, CLASS_INDEX)

  def get_frequency_by_attribute(self, data, attribute_index):
    frequency_map = {}
    for record in data:
      attr_val = record[attribute_index]
      frequency_map[attr_val] = frequency_map.get(attr_val, 0) + 1
    return frequency_map

  def is_numeric_attribute(self, attribute_index):
    return self.metadata._attributes[self.all_attributes[attribute_index]][0] == "numeric"

  def is_nominal_attribute(self, attribute_index):
      return attribute_index == CLASS_INDEX or self.metadata._attributes[self.all_attributes[attribute_index]][0] == "nominal"

def perform_classification():
  decision_tree = DecisionTree()
  decision_tree.learn(sys.argv[1], sys.argv[3], True, None)
  test_label_confidence_pairs = decision_tree.predict(sys.argv[2], True)
  return test_label_confidence_pairs
  

# def plot_learning_curve():
#   fractions = [0.05, 0.1, 0.2, 0.5]
#   avg_accuracy = {}
#   min_accuracy = {}
#   max_accuracy = {}
#   sizes = list()
#   for fraction in fractions:
#     total_accuracy_for_fraction = 0.0
#     for _ in range(10):
#       decision_tree = DecisionTree()
#       decision_tree.learn(sys.argv[1], 10, False, fraction)
#       decision_tree.predict(sys.argv[2], False)
#       size = decision_tree.dataset_size
#       current_accuracy = float(decision_tree.correct_predictions) / (decision_tree.correct_predictions + decision_tree.wrong_predictions)
#       if(not min_accuracy.has_key(size) or current_accuracy < min_accuracy[size]):
#         min_accuracy[size] = current_accuracy
#       if(not max_accuracy.has_key(size) or current_accuracy > max_accuracy[size]):
#         max_accuracy[size] = current_accuracy
#       total_accuracy_for_fraction += current_accuracy
#     avg_accuracy[size] = total_accuracy_for_fraction / 10.0
#     sizes.append(decision_tree.dataset_size)

#   # Average accuracy once for the entire dataset
#   decision_tree = DecisionTree()
#   decision_tree.learn(sys.argv[1], 10, False, None)
#   decision_tree.predict(sys.argv[2], False)
#   current_accuracy = float(decision_tree.correct_predictions) / (decision_tree.correct_predictions + decision_tree.wrong_predictions)
#   avg_accuracy[decision_tree.dataset_size] = min_accuracy[decision_tree.dataset_size] = max_accuracy[decision_tree.dataset_size] = current_accuracy
#   sizes.append(decision_tree.dataset_size)

#   sizes = sorted(sizes)
#   xs = []
#   ys = []
#   for size in sizes:
#     xs.append(size)
#     ys.append(min_accuracy[size])
#   plt.plot(xs, ys, 'k:', label='Minimum Accuracy')

#   xs = []
#   ys = []
#   for size in sizes:
#     xs.append(size)
#     ys.append(avg_accuracy[size])
#   plt.plot(xs, ys, 'k', label='Average Accuracy')

#   xs = []
#   ys = []
#   for size in sizes:
#     xs.append(size)
#     ys.append(max_accuracy[size])

#   plt.plot(xs, ys, 'k--', label='Maximum Accuracy')
#   legend = plt.legend(loc='lower right', shadow=True)
#   plt.title("Learning Curve")
#   plt.xlabel("Training Set Size")
#   plt.ylabel("Accuracy")
#   plt.show()

# def plot_roc_curve():
#   xs = []
#   ys = []
#   decision_tree = DecisionTree()
#   decision_tree.learn(sys.argv[1], sys.argv[3], False, None)
#   test_label_confidence_pairs = decision_tree.predict(sys.argv[2], False)
#   sorted_pairs = sorted(test_label_confidence_pairs, key = lambda row: (row[1], row[0]))
#   sorted_pairs = sorted_pairs[::-1]

#   labels = [record[0] for record in sorted_pairs]
#   num_neg = labels.count('-')
#   num_pos = labels.count('+')
#   tp = 0
#   fp = 0
#   last_tp = 0
#   fpr = 0
#   tpr = 0
#   xs.append(0.0)
#   ys.append(0.0)
#   for index in range(len(sorted_pairs)):
#     if index > 0 and sorted_pairs[index][1] != sorted_pairs[index - 1][1] and sorted_pairs[index][0] == '-' and tp > last_tp:
#       fpr = float(fp) / num_neg
#       tpr = float(tp) / num_pos
#       xs.append(fpr)
#       ys.append(tpr)
#       last_tp = tp
#     if sorted_pairs[index][0] == '+':
#       tp = tp + 1
#     else:
#       fp = fp + 1
#   xs.append(1.0)
#   ys.append(1.0)
#   plt.plot(xs, ys)
#   plt.title("ROC Curve")
#   plt.xlabel("False Positive Rate")
#   plt.ylabel("True Positive Rate")
#   plt.show()

if __name__ == "__main__":
  perform_classification()
  # plot_learning_curve()
  # plot_roc_curve()
