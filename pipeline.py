import time
import json
import sys
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from preprocess import process_data
from train_test import train, evaluate

args = sys.argv
if len(args) >= 3:
    roi = int(args[1])
    steps = int(args[2])
else:
    roi = 1
    steps = 5

print("\nRun pipeline with ROI #%d and %d time steps.\n" % (roi, steps))

# ===========================PROCESSING START===============================
processing_start = time.time()

X_Data, Y_Data = process_data(roi, steps)

enc = preprocessing.OneHotEncoder(categories='auto')
enc.fit(Y_Data)
Y_Data = enc.transform(Y_Data).toarray()

X_train, X_test, Y_train, Y_test = train_test_split(X_Data, Y_Data, test_size=0.1)

processing_end = time.time()
print("Preprocessed data in: %.2f sec\n" % (processing_end - processing_start))
# ===========================TRAINING START===============================

train_start = time.time()

model = train(X_train, Y_train)
# save_model(model, "model.json")
evaluate(model, X_test, Y_test)

train_end = time.time()
print("Train and test in: %.2f sec" % (train_end - train_start))
# ===========================TRAINING END===============================

result = {
    "processing_start": processing_start,
    "processing_end": processing_end,
    "train_start": train_start,
    "train_end": train_end
}

with open("timestamps.json", "w") as outfile:
    json.dump(result, outfile)
