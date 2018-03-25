import math
def sigmoid(value):
  denominator = 1.0 + math.exp(-value)
  numerator = 1.0
  return numerator/denominator

if __name__ == "__main__":
  print sigmoid(1 + 3*sigmoid(9) + 2*sigmoid(17) + sigmoid(-3))