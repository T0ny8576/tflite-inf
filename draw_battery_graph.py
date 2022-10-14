import os
import numpy as np
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FOLDER = os.path.join(PROJECT_DIR, "logs")
OUTPUT_FOLDER = os.path.join(PROJECT_DIR, "plots")


def draw(log_file, output_file, label):
    output_path = os.path.join(OUTPUT_FOLDER, output_file)
    with open(os.path.join(LOG_FOLDER, log_file), "r") as fi:
        lines = fi.readlines()
        timestamps = []
        events = []
        start_time = 0
        stop_time = 0
        total_images = None
        unique_images = None
        for line in lines:
            if "Time: " in line:
                fields = line.split("\t")
                timestamps.append(int(fields[0].split(" ")[-1]))
                events.append(fields[1].strip())
            elif "Total Images: " in line:
                total_images = int(line.split(" ")[-1].strip())
            elif "Unique Images: " in line:
                unique_images = int(line.split(" ")[-1].strip())
            elif "Start: " in line:
                start_time = int(line.split(" ")[-1].strip())
            elif "Stop: " in line:
                stop_time = int(line.split(" ")[-1].strip())
        total_time = stop_time - start_time

        voltage_t = []
        current_t = []
        voltage_y = []
        current_y = []
        for i, event in enumerate(events):
            if timestamps[i] < start_time:
                continue
            if "Battery voltage" in event:
                voltage_t.append(timestamps[i] - start_time)
                voltage_y.append(int(event.split(" ")[-1]))
            elif "Current" in event:
                current_t.append(timestamps[i] - start_time)
                current_y.append(int(event.split(" ")[-1]))

        voltage_y = np.asarray(voltage_y) / 1000.
        current_y = np.asarray(current_y) / 1000000.
        voltage_t = np.asarray(voltage_t) / 1000.
        current_t = np.asarray(current_t) / 1000.

        fig, ax = plt.subplots()
        ax.plot(current_t, current_y, color="red", marker=".")
        ax.set_xlabel("time /s", fontsize=12)
        ax.set_ylabel("current /A", color="red", fontsize=12)
        ax.set_ylim((0, 1))

        ax2 = ax.twinx()
        ax2.plot(voltage_t, voltage_y, color="blue", marker=".")
        ax2.set_ylabel("voltage /V", color="blue", fontsize=12)
        ax2.set_ylim((0, 5))
        plt.title(label)
        plt.show()
        fig.savefig(output_path, format='jpeg', dpi=100, bbox_inches='tight')

        power_y = []
        vi = 0
        for ci, ts in enumerate(current_t):
            while vi + 1 < len(voltage_t) and voltage_t[vi + 1] <= ts:
                vi += 1
            power_y.append(voltage_y[vi] * current_y[ci])
        return label, current_t, power_y, total_time, total_images, unique_images


def compare_power(profiles, output_file):
    output_path = os.path.join(OUTPUT_FOLDER, output_file)
    fig, ax = plt.subplots()
    ax.set_xlabel("time /s", fontsize=12)
    ax.set_ylabel("power /W", fontsize=12)
    ax.set_ylim((0, 4))

    for prof in profiles:
        print(prof[0] + ":")
        ax.plot(prof[1], prof[2], marker=".", label=prof[0])
        time_per_frame = prof[3] / prof[4]
        print("Total images: {}".format(prof[4]))
        if prof[5] is not None:
            print("Unique images: {}".format(prof[5]))
            print("Removal rate: {:.2%}".format(1. - prof[5] / prof[4]))
        print("Total time (ms): {}".format(prof[3]))
        print("Time per frame (ms): {}".format(time_per_frame))
        average_power = np.sum(prof[2]) / len(prof[2])
        print("Average power (W): {}".format(average_power))
        energy_per_frame = average_power * time_per_frame / 1000.
        print("Energy per frame (J): {}".format(energy_per_frame))
        print()

    plt.title("Power Comparison")
    plt.legend()
    plt.show()
    fig.savefig(output_path, format='jpeg', dpi=100, bbox_inches='tight')


if __name__ == "__main__":
    # thumbsup_profile = draw("TFLTest-thumbsup-1s.txt", "battery_thumbsup.jpg", "Thumbs-up Detection")
    # od_profile = draw("TFLTest-ed0-1s.txt", "battery_od.jpg", "EfficientDet D0")
    # od_classifier_profile = draw("TFLTest-ed0-r50-1s.txt", "battery_od_classifier.jpg", "ED0 + ResNet50")
    # phash_profile = draw("TFLTest-phash-1s.txt", "battery_phash.jpg", "Perceptual Hash")
    # phash_od_profile = draw("TFLTest-phash-ed0-1s.txt", "battery_phash_od.jpg", "pHash + ED0")
    # phash_od_classifier_profile = draw("TFLTest-phash-ed0-r50-1s.txt", "battery_phash_od_classifier.jpg",
    #                                    "pHash + ED0 + ResNet50")
    # power_profiles = [thumbsup_profile, od_profile, od_classifier_profile,
    #                   phash_profile, phash_od_profile, phash_od_classifier_profile]

    echo_cloudlet_profile = draw("TFLTest-600-echo-1s.txt", "battery_echo_cloudlet.jpg", "Echo Cloudlet")
    od_cloudlet_classifier_profile = draw("TFLTest-600-ed0-mpn-1s.txt", "battery_od_cloudlet_classifier.jpg",
                                          "ED0 + Cloudlet MPN COV")
    cloudlet_od_classifier_profile = draw("TFLTest-600-frcnn-mpn-1s.txt", "battery_cloudlet_od_classifier.jpg",
                                          "Cloudlet FRCNN + Cloudlet MPN COV")
    power_profiles = [echo_cloudlet_profile, od_cloudlet_classifier_profile, cloudlet_od_classifier_profile]

    compare_power(power_profiles, "power_comparison.jpg")
