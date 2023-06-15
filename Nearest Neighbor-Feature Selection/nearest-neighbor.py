import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# calculate the eucledian distance between 2 numpy arrays
def calculateDistance(train, test):
  return np.sqrt(np.sum((train - test) ** 2, axis = 1))

# function for the forward selection technique which selects the feature that provides the greatest improvement is selected and added to the feature set in each iteration
def forwardSelection(X,y):
  selectedFeatures = []
  localBestFeatures = []
  bestAccuracy = 0
  n = len(X[0])

  f =

  # For loop for level of forward selection
  for i in range(n):

    # We can add a condition here to stop and return the best result calculated so far. This condition can be time/accuracy/#iterations threshold.

    bestFeature = None
    accuracy = 0
    localBest = 0

    # This for loop calculates which feature is the best feature at this level
    for feature in range(n):
      if feature in localBestFeatures:
        continue
      accuracy = leaveOneOut(X,y,localBestFeatures+[feature])
      print(f"Using feature(s) {localBestFeatures+[feature]} accuracy is {accuracy*100:.4f}%")
      if accuracy > localBest:
        localBest = accuracy
        bestFeature = feature

    localBestFeatures.append(bestFeature)

    # If accuracy for this level beats the global accuracy, we update the selected feature set and global accuracy
    if localBest > bestAccuracy:
      bestAccuracy = localBest
      selectedFeatures = deepcopy(localBestFeatures)
      print(f"Feature set {selectedFeatures} was best for this iteration with an accuracy of {bestAccuracy*100:.4f}%\n")
    else:
      print(f"Warning, accuracy has decreased! Feature set {localBestFeatures} was local best for this iteration with an accuracy of {localBest*100:.4f}%")
      print(f"Continuing search in case of local maxima\n")


  print(f"Search finished! The best feature set is {selectedFeatures} with an accuracy of {bestAccuracy*100:.4f}%")
  return selectedFeatures, bestAccuracy

def backwardElimination(X, y):
  selectedFeatures = list(range(len(X[0])))
  localBestFeatures = list(range(len(X[0])))
  bestAccuracy = leaveOneOut(X, y, selectedFeatures)

  # While loop for the level of backward elimination
  while len(localBestFeatures) > 1:

    # We can add a condition here to stop and return the best result calculated so far. This condition can be time/accuracy/#iterations threshold.

    worstFeature = None
    localBest = 0

    # This for loop calculates which feature is the worst feature at this level
    for feature in localBestFeatures:
      currentFeatures = deepcopy(localBestFeatures)
      currentFeatures.remove(feature)
      accuracy = leaveOneOut(X, y, currentFeatures)
      print(f"Using feature(s) {currentFeatures} accuracy is {accuracy*100:.4f}%")
      if accuracy > localBest:
        localBest = accuracy
        worstFeature = feature

    localBestFeatures.remove(worstFeature)

    # If accuracy for this level beats the global accuracy, we update the selected feature set and global accuracy
    if localBest > bestAccuracy:
      selectedFeatures.remove(worstFeature)
      bestAccuracy = localBest
      selectedFeatures = deepcopy(localBestFeatures)
      print(f"Feature set {selectedFeatures} was best for this iteration with an accuracy of {bestAccuracy*100:.4f}, removed feature is {worstFeature}\n")
    else:
      print(f"Warning, accuracy has decreased! Feature set {localBestFeatures} was local best for this iteration with an accuracy of {localBest*100:.4f}%")
      print(f"Removed feature is {worstFeature}, Continuing search in case of local maxima\n")
  print(f"Search finished! The best feature set is {selectedFeatures} with an accuracy of {bestAccuracy*100:.4f}%")
  return selectedFeatures, bestAccuracy

# Leave one out evaluation function used with Nearest Neighbours
def leaveOneOut(X, y, selectedFeatures=[]):
  rows = len(X)
  count = 0

  # i-th row is the test data
  for i in range(rows):
    minDistance = float('inf')
    X_temp = np.delete(X,i,axis=0)

    # Calculating distances considering i-th row as test data
    dist = calculateDistance(X[i,selectedFeatures], X_temp[:,selectedFeatures])

    # Getting the index of the minimum distance and incrementing count if classified correctly
    index = np.argmin(dist,keepdims=True)
    if i <= index:
      if y[i] == y[index+1]:
        count += 1
    else:
      if y[i] == y[index]:
        count += 1

  accuracy = count/rows
  return accuracy

def preprocess(data):
  # Spliting labels and train data
  X, y = data[1:, :-1], data[1:, -1]

  # Converting class labels to float
  y = y.astype(int)

  # Converting train data features to float
  columns_to_convert = [0,3,4,5,7,9]
  X = X.astype(object)
  X[:, columns_to_convert] = X[:, columns_to_convert].astype(float)

  # Converting categorical features to numerical data
  le = preprocessing.LabelEncoder()
  X_pd = pd.DataFrame(X)
  X_pd[[1,2,6,8,10]] = le.fit_transform(X_pd[[1,2,6,8,10]].values.reshape(-1,1)).reshape(-1,5)
  X = X_pd.values

  # Normalizing the data using StandardScaler()
  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  return X, y

print("\n------------------------------------------------------------\n")
print("\tWelcome to the feature selection algorithm!")
print("\n------------------------------------------------------------\n")
while True:
  print("\nChoose the dataset you want to run the algorithm for:\n 1. Small\n 2. Large\n 3. XXXLarge\n 4. Real world")
  choice = int(input().strip())
  if choice == 1:
    filename = "CS170_small_Data__8.txt"
    break
  elif choice == 2:
    filename = "CS170_large_Data__26.txt"
    break
  elif choice == 3:
    filename = "CS170_XXXlarge_Data__11.txt"
    break
  elif choice == 4:
    filename = "heart.csv"
    break
  else:
    print("Wrong choice. Please select valid options")

if choice == 4:
  data = np.loadtxt("heart.csv", delimiter=",", dtype=str)
  X, y = preprocess(data)
else:
  data = np.loadtxt(filename, dtype=float)
  X, y = data[:, 1:], data[:, 0]

print(f"The dataset has {len(X[0])} features (not including class label) and {len(X)} instances")
selectedFeatures = list(range(len(X[0])))
allAccurcy = leaveOneOut(X, y, selectedFeatures)
print(f"Running Nearest Neighbours with all {len(X[0])} features, using leaving one out evaluation, we get an accuracy of {allAccurcy*100:.4f}%")
print("Choose the algorithm you want to run:\n 1. Forward Selection \n 2. Backward Elimination \n")
choice = int(input().strip())
if choice == 1:
    print("Begin search using forward selection!\n")
    forwardSelection(X, y)
elif choice == 2:
    print("Begin search using backward elimination!\n")
    backwardElimination(X, y)
else:
  print("Wrong choice!")
