import time
import json
import sys
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from preprocess import process_data
from train_test import train, evaluate, save_model, load_model

args = sys.argv
if len(args) >= 3:
    subject = int(args[1])
    steps = int(args[2])
else:
    subject = 4
    steps = 5

print("\nRun pipeline with ROI #%d and %d time steps.\n" % (subject, steps))


def run(subject, steps):
    # ===========================PROCESSING START===============================
    processing_start = time.time()

    X_Data, Y_Data = process_data(subject, steps)

    enc = preprocessing.OneHotEncoder(categories='auto')
    enc.fit(Y_Data)
    Y_Data = enc.transform(Y_Data).toarray()

    X_train, X_test, Y_train, Y_test = train_test_split(X_Data, Y_Data, test_size=0.25)

    processing_end = time.time()
    print("Preprocessed data in: %.2f sec\n" % (processing_end - processing_start))
    # ===========================TRAINING START===============================

    # train_start = time.time()

    model = train(X_train, Y_train)
    # save_model(model, "model_roi1_6steps.json")
    # model = load_model("model_roi1_5steps.json")
    f1, acc = evaluate(model, X_test, Y_test)

    # train_end = time.time()
    # print("Train and test in: %.2f sec" % (train_end - train_start))

    # ===========================TRAINING END===============================

    # result = {
    #     "processing_start": processing_start,
    #     "processing_end": processing_end,
    #     "train_start": train_start,
    #     "train_end": train_end
    # }
    #
    # with open("timestamps.json", "w") as outfile:
    #     json.dump(result, outfile)

    return subject, steps, f1, acc


tasks = [(4, 5), (4, 2), (3, 5), (3, 2), (2, 5), (2, 2), (1, 5), (1, 2)]
result = []
for task in tasks:
    result.append(run(task[0], task[1]))

for item in result:
    print("sub=%d, step=%d, f1=%.4f, acc=%.4f " % item)
