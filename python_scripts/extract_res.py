import argparse
import os

# Create a command-line argument parser
parser = argparse.ArgumentParser(
    description="Convert and merge log files to desired format"
)
parser.add_argument("log_dir", type=str, help="Paths to the input log files")
parser.add_argument("--prefix", type=str)
args = parser.parse_args()

dirs = os.listdir(args.log_dir)

# Initialize a dictionary to store the success rates for the current file
success_rates = {}

# Iterate through the list of input log files
for log_dir in dirs:
    if ".DS_Store" in log_dir:
        continue
    if not log_dir.startswith(args.prefix):
        continue
    if log_dir.split("-seed")[0] != args.prefix:
        continue

    seed = int(log_dir.split("-seed")[-1])
    log_dir = os.path.join(args.log_dir, log_dir)
    tmp_dir = list(os.listdir(log_dir))[0]
    log_dir = os.path.join(log_dir, tmp_dir)
    log_file = os.path.join(log_dir, "test", f'{tmp_dir.replace("_eval", "-eval")}.log')

    with open(log_file, "r") as file:
        lines = file.readlines()

    success_rate = None

    # Iterate through the lines in the current log file
    for line in lines:
        if "Success or Early Stop Rate:" in line:
            success_rate = line.split(" ")[-1].strip()

    success_rates[seed] = success_rate

# Print the results
# for seed, success_rate in success_rates.items():
#     print(f"{seed}: {success_rate}")s
print(f"{0}: {success_rates[0]}")
print(f"{17898679}: {success_rates[17898679]}")
print(f"{829384}: {success_rates[829384]}")
print(f"{33794}: {success_rates[33794]}")
print(f"{94}: {success_rates[94]}")
print(f"{26272}: {success_rates[26272]}")
