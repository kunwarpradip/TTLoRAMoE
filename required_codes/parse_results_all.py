import os
import json
import csv
import argparse
import concurrent.futures

def process_directory(directory, base_path):
    item_path = os.path.join(base_path, directory)
    result_path = os.path.join(item_path, 'result.json')

    if os.path.exists(result_path):
        try:
            with open(result_path, 'r') as result_file:
                result_content = json.load(result_file)

            if result_content:
                return (directory, result_content)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error decoding JSON in directory {directory}: {e}")

    return None

def main(data_type):
    # Define the base path to the directory containing the subdirectories
    base_path = f'/usr/projects/unsupgan/afia/ray_tune_roberta/tune_roberta'
    # base_path = f'/home/user/Desktop/LLMs/TTLoRAMoE/Running_TTLoRA'

    # List all directories in the base path
    directories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    # Use ThreadPoolExecutor to process directories in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda d: process_directory(d, base_path), directories))

    # Define the CSV file name
    output_file = f'results_{data_type}.csv'

    # Write the results to the CSV file
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = [
            'directory', 'val_acc', 'trainable_params', 'timestamp', 'checkpoint_dir_name', 'done',
            'training_iteration', 'trial_id', 'date', 'time_this_iter_s', 'time_total_s', 'pid',
            'hostname', 'node_ip', 'config_shapes', 'config_ranks', 'config_alpha', 'config_learning_rate',
            'time_since_restore', 'iterations_since_restore'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            if result:
                directory, result_content = result
                config = result_content.get('config', {})
                writer.writerow({
                    'directory': directory,
                    'val_acc': result_content.get('val_acc'),
                    'trainable_params': result_content.get('trainable_params'),
                    'timestamp': result_content.get('timestamp'),
                    'checkpoint_dir_name': result_content.get('checkpoint_dir_name'),
                    'done': result_content.get('done'),
                    'training_iteration': result_content.get('training_iteration'),
                    'trial_id': result_content.get('trial_id'),
                    'date': result_content.get('date'),
                    'time_this_iter_s': result_content.get('time_this_iter_s'),
                    'time_total_s': result_content.get('time_total_s'),
                    'pid': result_content.get('pid'),
                    'hostname': result_content.get('hostname'),
                    'node_ip': result_content.get('node_ip'),
                    'config_shapes': ','.join(map(str, config.get('shapes', []))),
                    'config_ranks': config.get('ranks'),
                    'config_alpha': config.get('alpha'),
                    'config_learning_rate': config.get('learning_rate'),
                    'time_since_restore': result_content.get('time_since_restore'),
                    'iterations_since_restore': result_content.get('iterations_since_restore')
                })

    print(f"Contents have been written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and collect results from specified directories.')
    parser.add_argument('data_type', type=str, choices=['cb', 'wsc', 'copa', 'boolq','all'], help='The data type subdirectory to process.')

    args = parser.parse_args()
    main(args.data_type)

