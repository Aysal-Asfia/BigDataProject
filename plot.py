import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv


def plot_summary(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        subject = []
        data_size = []
        no_of_exp = []
        no_of_features = []
        mem_used_5 = []
        mem_used_2 = []
        processing_5 = []
        processing_2 = []
        train_5 = []
        train_2 = []
        f1_5 = []
        f1_2 = []
        acc_5 = []
        acc_2 = []

        for line in csv_reader:
            subject.append(float(line[0]))
            data_size.append(int(line[1]))
            no_of_exp.append(int(line[2]))
            no_of_features.append(int(line[3]))
            mem_used_5.append(int(line[4]))
            mem_used_2.append(int(line[5]))
            processing_5.append(float(line[6]))
            processing_2.append(float(line[7]))
            train_5.append(float(line[8]))
            train_2.append(float(line[9]))
            f1_5.append(float(line[10]))
            f1_2.append(float(line[11]))
            acc_5.append(float(line[12]))
            acc_2.append(float(line[13]))

        group_names = ["Sub#1", "Sub#2", "Sub#3", "Sub#4"]

        # df = pd.DataFrame(dict(processing_5=processing_5, processing_2=processing_5, train_5=train_5, train_2=train_2), group_names)

        # fig, ax = plt.subplots()
        # df[["processing_5", "train_5"]].plot.bar(stacked=True, position=1.5, width=.2, ax=ax, color="gc", alpha=0.5)
        # df[["processing_2", "train_2"]].plot.bar(stacked=True, position=0.5, width=.2, ax=ax, color="ry", alpha=0.5)
        # plt.legend(["processing time TR1-5", "training time TR1-5", "processing 6time TR34", "training time TR34"])
        # plt.xticks(rotation=0)
        # plt.ylabel("time (s)")
        # plt.show()

        figure = plt.figure()
        ax1 = figure.add_subplot(1, 2, 1)
        ax1.bar(np.arange(4), f1_5, 0.3, label="TR1-5")
        ax1.bar(np.arange(4) + 0.3, f1_2, 0.3, label="TR34")
        ax1.set_title("f1 score")
        ax1.set_ylim(bottom=0, top=1)
        ax1.legend(loc="upper left")

        ax2 = figure.add_subplot(1, 2, 2, sharex=ax1)
        ax2.bar(np.arange(4), acc_5, 0.3, label="TR1-5")
        ax2.bar(np.arange(4) + 0.3, acc_2, 0.3, label="TR34")
        ax2.set_title("accuracy")
        ax2.set_ylim(bottom=0, top=1)
        # plt.bar(group_names, train_5, 0.2, bottom=processing_5, label="training time TR1-5")
        # plt.bar(group_names, data_size, mem_used_2, color='g', linewidth=1, linestyle=":", label="memory used TR34")
        # plt.bar(group_names, processing_2, 0.2, label="processing time TR34")
        # plt.bar(group_names, train_2, 0.2, bottom=processing_2, label="training time TR34")
        # plt.axis([200, 1000, 0, 3000])
        plt.xticks(range(len(group_names)), group_names)
        plt.show()


plot_summary("summary_update.csv")
