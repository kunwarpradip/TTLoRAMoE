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
    base_path = f'/usr/projects/unsupgan/afia/ray_tune_roberta/tune_roberta_{data_type}'
    # base_path = f'/home/user/Desktop/LLMs/TTLoRAMoE/Running_TTLoRA{data_type}'

    # List all directories in the base path
    directories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    # Use ThreadPoolExecutor to process directories in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda d: process_directory(d, base_path), directories))

    # Filter out None results and sort by val_acc in descending order
    results = [result for result in results if result]
    results.sort(key=lambda x: x[1].get('val_acc', 0), reverse=True)

    # Define the CSV file name
    output_file = f'results_{data_type}.csv'

    # Write the results to the CSV file
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = [
            'val_acc', 'trainable_params', 'training_iteration', 'time_this_iter_s',
            'time_total_s', 'config_shapes', 'config_ranks', 'config_alpha', 'config_learning_rate'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            _, result_content = result
            config = result_content.get('config', {})
            writer.writerow({
                'val_acc': result_content.get('val_acc'),
                'trainable_params': result_content.get('trainable_params'),
                'training_iteration': result_content.get('training_iteration'),
                'time_this_iter_s': result_content.get('time_this_iter_s'),
                'time_total_s': result_content.get('time_total_s'),
                'config_shapes': ','.join(map(str, config.get('shapes', []))),
                'config_ranks': config.get('ranks'),
                'config_alpha': config.get('alpha'),
                'config_learning_rate': config.get('learning_rate')
            })

    print(f"Contents have been written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and collect results from specified directories.')
    parser.add_argument('data_type', type=str, help='The data type subdirectory to process.')

    args = parser.parse_args()
    main(args.data_type)

