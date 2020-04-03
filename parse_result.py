import json

import matplotlib.pyplot as plt
import numpy as np

timestamps_file = "log/roi1/5steps/timestamps.json"
mem_prof_file   = "log/roi1/5steps/atop.log"
collectl_file   = "log/roi1/5steps/collectl-simgrid-vm-20200402.dsk.csv"

f = open(mem_prof_file)
lines = f.readlines()

sys_mem = []
free_mem = []
used_mem = []
# app_mem = []
cache_used = []
avai_mem = []
dirty_ratio = []
dirty_bg_ratio = []
dirty_data = []

swap_size = []
swap_free = []

# bw_r = []
# bw_w = []
for i in range(len(lines)):
    line = lines[i]
    if line.startswith("MEM"):
        values = line.split(" ")

        sys_mem_mb = int(values[7]) * 4096 / 2 ** 20
        sys_mem.append(sys_mem_mb)

        free_mem_mb = int(values[8]) * 4096 / 2 ** 20
        free_mem.append(free_mem_mb)

        used_mem.append(sys_mem_mb - free_mem_mb)

        cache_in_mb = int(values[9]) * 4096 / 2 ** 20
        cache_used.append(cache_in_mb)

        dirty_amt_mb = int(values[12]) * 4096 / 2 ** 20
        dirty_data.append(dirty_amt_mb)

        available_mb = free_mem_mb + cache_in_mb - dirty_amt_mb
        avai_mem.append(available_mb)

        dirty_ratio.append(0.2 * available_mb)
        dirty_bg_ratio.append(0.1 * available_mb)

    else:
        if line.startswith("SWP"):
            values = line.split(" ")
            swap_size.append(int(values[7]) * 4096 / 2 ** 20)
            swap_free.append(int(values[8]) * 4096 / 2 ** 20)

intervals = len(dirty_data)
dirty_data = np.array(dirty_data)
time = np.arange(0, intervals)

# ==========================MEM PROFILING===================================
with open(timestamps_file) as time_stamp_file:
    timestamps = json.load(time_stamp_file)


def timestamp_plot(fig, time_stamps):
    processing_start = time_stamps["processing_start"]
    processing_end = time_stamps["processing_end"]
    train_start = time_stamps["train_start"]
    train_end = time_stamps["train_end"]

    fig.axvspan(xmin=processing_end - processing_start, xmax=processing_start - processing_start, color="g",
                alpha=0.2, label="data processing")
    fig.axvspan(xmin=train_start - processing_start, xmax=train_end - processing_start, color="b",
                alpha=0.2, label="train/test")


def mem_plot(fig, readonly=False):
    fig.minorticks_on()
    fig.set_title("memory profiling (ROI #1, no_steps = 5)")
    timestamp_plot(fig, timestamps)

    # app_cache = list(np.array(app_mem) + np.array(cache_used))

    fig.plot(time, sys_mem, color='k', linewidth=1.5, label="total mem")
    # ax1.plot(time, free_mem, color='g', linewidth=1.5, linestyle="-.", label="free memory")
    fig.plot(time, used_mem, color='g', linewidth=1.5, label="used mem")
    # ax1.plot(time, app_mem, color='c', linewidth=1.5, label="app memory")
    fig.plot(time, cache_used, color='m', linewidth=1.5, label="cache used")
    # ax1.plot(time, app_cache, color='c', linewidth=1.5, label="cache + app")
    # fig.plot(time, dirty_data, color='r', linewidth=1.5, label="dirty data")
    fig.plot(time, avai_mem, color='b', linewidth=1, linestyle="-.", label="available mem")
    # fig.plot(time, dirty_ratio, color='k', linewidth=1, linestyle="-.", label="dirty_ratio")
    # fig.plot(time, dirty_bg_ratio, color='r', linewidth=1, linestyle="-.", label="dirty_bg_ratio")

    fig.set_ylabel("memory (MB)")
    fig.legend(fontsize='small', loc='best')


def collectl_plot(fig, readonly=False):
    dsk_data = np.loadtxt(collectl_file, skiprows=1, delimiter=',')
    read = dsk_data[:, 2] / 1024
    write = dsk_data[:, 6] / 1024

    time = np.arange(0, len(read))

    fig.minorticks_on()
    fig.set_title("disk throughput (MB)")

    timestamp_plot(fig, timestamps)

    fig.plot(time, read, color='g', linewidth=1.5, label="read bw")
    fig.plot(time, write, color='r', linewidth=1.5, label="write bw")

    fig.set_xlabel("time (s)")
    fig.set_ylabel("memory (MB)")
    fig.legend(fontsize='small', loc='best')


figure = plt.figure()
plt.tight_layout()
ax1 = figure.add_subplot(2, 1, 1)
ax2 = figure.add_subplot(2, 1, 2, sharex=ax1)

ax1.set_ylim(top=16000, bottom=-1000)
ax1.set_xlim(left=-10, right=300)
mem_plot(ax1)

collectl_plot(ax2)

plt.show()
