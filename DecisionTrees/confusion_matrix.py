import sys
import csv

def build_confusion_matrix(file):
  matrix = [[0 for x in range(3)] for y in range(3)] 
  
  with open(file, 'rb') as csvfile:
    records = csv.reader(csvfile)
    for record in records:
      row = int(record[0]) - 1
      col = int(record[1]) - 1
      matrix[row][col] = matrix[row][col] + 1;
  print matrix

if __name__ == "__main__":
  build_confusion_matrix(sys.argv[1])