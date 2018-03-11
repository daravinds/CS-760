import matplotlib.pyplot as plt


# This code is a READONLY version for plotting PR curve for Naive Bayes and TAN.
# Cannot be run standalone. Can be run only from bayes.py only
# by uncommenting the calls to plot_precision_recall_curve method

def plot_precision_recall_curve(curve_data, positive_label, bayes):
  curve_data.sort(key=lambda x: (-x[2]))
  total_positive_labels = 0
  for values in curve_data:
    if values[0] == positive_label:
      total_positive_labels += 1

  true_positives_so_far = 0
  xs = list()
  ys = list()
  for index, record in enumerate(curve_data):
    if record[0] == positive_label:
      true_positives_so_far += 1
    precision = float(true_positives_so_far) / (index + 1.0)
    recall = float(true_positives_so_far) / (total_positive_labels)
    xs.append(recall)
    ys.append(precision)

  if bayes:
    curve_type = "Naive Bayes"
  else:
    curve_type = "Tree Augmented Network"
  plt.axis([0, 1.2, 0, 1.2])
  plt.title(str(curve_type) + " Precision-Recall Curve")
  plt.xlabel("Recall")
  plt.ylabel("Precision")
  plt.plot(xs, ys)
  plt.show()