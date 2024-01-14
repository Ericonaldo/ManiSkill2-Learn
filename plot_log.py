import re
import matplotlib.pyplot as plt
import argparse
import os


def plot_training_curve(log_file):
    step_pattern = r"(\d+)/\d+"
    # loss_pattern = r"action_diff_loss: (\d+\.\d+)"
    loss_pattern = r"keyframe_loss: (\d+(?:\.\d+)?(?:e[-+]?\d+)?)"
    steps = []
    losses = []

    with open(log_file, "r") as file:
        log_entries = file.read()

        matches = re.findall(step_pattern + r".*?" + loss_pattern, log_entries)
        for match in matches:
            step = int(match[0])
            loss = float(match[1])
            steps.append(step)
            losses.append(loss)

    plt.plot(steps, losses)
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.title("Training Curve")
    plt.grid(True)
    # plt.show()

    # Set a non-uniform y-axis scale
    plt.yscale("symlog", linthresh=0.01)

    # Save the plot with the same name as the log file
    plot_filename = os.path.splitext(log_file)[0] + ".png"
    plt.savefig(plot_filename)
    plt.close()

    print(f"Training curve plot saved as {plot_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curve from log file")
    parser.add_argument("log_file", type=str, help="Path to the log file")
    args = parser.parse_args()

    plot_training_curve(args.log_file)
