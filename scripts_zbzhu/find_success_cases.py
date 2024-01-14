import csv


if __name__ == "__main__":
    diff_log_dir = "complete_logs/StackCube-v0/KeyDiffAgent/posediff-quat-keydiff90000-seed94/20231014_052651_eval/test"
    kf_diff_log_dir = "complete_logs/StackCube-v0/KeyDiffAgent/posediff-quat-diff90000-statekey200000-seed94/20231014_071434_eval/test"

    # read data from csv file
    csv_file = open(diff_log_dir + "/statistics.csv", "r")
    reader = csv.reader(csv_file)
    diff_data = list(reader)
    csv_file.close()
    diff_successes = [int(float(d[2])) for d in diff_data[1:]]

    # read data from csv file
    csv_file = open(kf_diff_log_dir + "/statistics.csv", "r")
    reader = csv.reader(csv_file)
    kf_diff_data = list(reader)
    csv_file.close()
    kf_diff_successes = [int(float(d[2])) for d in kf_diff_data[1:]]

    for i in range(len(diff_successes)):
        if diff_successes[i] == 0 and kf_diff_successes[i] == 1:
            print(i)
