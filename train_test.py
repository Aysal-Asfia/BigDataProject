import numpy as np
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, LSTM
from keras.models import Sequential, model_from_json
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt


def create_lstm_model(no_steps, no_features):
    model = Sequential()

    # input_data shape = (num_trials, timesteps, input_dim)
    ## activation function:  tanh , relu,...
    model.add(LSTM(500, input_shape=(no_steps, no_features), return_sequences=True, activation='relu', dropout=0.2))
    # returns a sequence of vectors of dimension 60
    model.add(LSTM(250, activation='tanh', dropout=0.2, return_sequences=True))
    # returns a sequence of vectors of dimension 60
    model.add(LSTM(100, activation='tanh', dropout=0.2))
    # return a single vector of dimension 40
    model.add(Dense(5, activation="softmax"))  # number of classes 3, object categori,....

    adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


def train(X_train, Y_train):

    _, no_steps, no_features = X_train.shape
    # training:
    # out_data shape = (num_trials, num_classes)
    batch_size = 50  # 10,20,....
    epochs = 50  # 20,30,...
    model = create_lstm_model(no_steps, no_features)

    ## monitor accuracy and save the best models; it seems not much improve after 33 epochs
    filepath = "./weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit(X_train, Y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list,
              verbose=0)

    return model


def evaluate(model, X_test, Y_test):

    # Accuracy
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # ROC curve
    Y_predict = model.predict(X_test)
    roc_auc_score(Y_test, Y_predict)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(5):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), Y_predict.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()

    M = model.predict_classes(X_test)

    ytt = np.zeros((Y_test.shape[0], 1))
    for i, val in enumerate(Y_test):
        val = val.tolist()
        ytt[i] = int(val.index(1.0))

    report = classification_report(ytt, M)
    print("classification report")
    print(report)

    matrix = confusion_matrix(ytt, M)
    print("confusion matrix")
    print(matrix)

    plt.imshow(matrix, interpolation=None, cmap='binary')
    plt.colorbar()
    plt.title('confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def save_model(model, model_file):
    model_json = model.to_json()
    with open(model_file, "w") as json_file:
        json_file.write(model_json)
    model.save_weights("%s.h5" % model_file)


def load_model(model_file):
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("%s.h5" % model_file)

    adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    loaded_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return loaded_model

