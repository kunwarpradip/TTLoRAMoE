import os
import re
import csv

# Define the path to the directory containing the log files
#log_path = '/usr/projects/unsupgan/afia/ray_tune_llama2/classification/job_logs_v2'  # Adjust this path as needed
log_path = '/usr/projects/unsupgan/afia/ray_tune_roberta/job_logs_v2'
# log_path= '/home/user/Desktop/LLMs/TTLoRAMoE/Running_TTLoRA/logs'

# Define the output CSV file name
output_file = 'results_summary_deberta.csv'

# Define a regex pattern to extract relevant information from the log lines
pattern = re.compile(
    r"Current best trial: \S+ with val_acc=(?P<val_acc>[\d\.]+) and params={'shapes': (?P<shapes>\[.*?\]), 'ranks': (?P<ranks>\d+), 'alpha': (?P<alpha>\d+), 'learning_rate': (?P<learning_rate>[\d\.]+)}"
)

# Function to process a single file
def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            match = pattern.search(line)
            if match:
                return match.groupdict()
    return None

# List to hold the result rows
results = []

# Iterate through all .out files in the directory
for file_name in os.listdir(log_path):
    if file_name.endswith('.out'):
        data_type = file_name.split('_')[0]
        file_path = os.path.join(log_path, file_name)
        result = process_file(file_path)
        if result:
            result['data'] = data_type
            results.append(result)

# Write the results to the CSV file
with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['data', 'shapes', 'ranks', 'alpha', 'learning_rate', 'val_acc']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"Results have been written to {output_file}")

