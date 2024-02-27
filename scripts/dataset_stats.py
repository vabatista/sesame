## This script collect some statistics from the dataset files.

import json
import os
import statistics

# Initialize variables


# Loop through all JSONL files in current directory
for filename in os.listdir():
    if filename.endswith('.jsonl'):
        context_sizes = []
        qas_sizes = []
        ans_sizes = []
        with open(filename, 'r') as f:
            # Loop through each line in the file
            for line in f:
                # Load JSON object from line
                obj = json.loads(line)
                # Check if "context" attribute is present
                if 'context' in obj:
                    # Add size of "context" attribute to list
                    context_sizes.append(len(obj['context']))
                if 'qas' in obj:
                    qas_sizes.append(len(obj['qas']))
                    for qa in obj['qas']:
                        if 'answers' in qa:
                            if type(qa['answers'][0]) == dict:
                                ans_sizes.append(len(qa['answers'][0]['text']))
                            else:
                                ans_sizes.append(len(qa['answers'][0]))
        # Calculate basic statistics
        mean_size = statistics.mean(context_sizes)
        max_size = max(context_sizes)
        std_dev = statistics.stdev(context_sizes)

        mean_size_qa = statistics.mean(qas_sizes)
        max_size_qa = max(qas_sizes)
        std_dev_qa = statistics.stdev(qas_sizes)

        # Print summary
        print(f"Stats for Dataset: {filename}")
        print(f"Total triples: {sum(qas_sizes)}")
        print(f"context' size: {len(context_sizes)} mean: {mean_size:.2f} Max size of: {max_size}  Std: {std_dev:.2f}")
        print(f"qas' size: {len(context_sizes)} mean: {mean_size_qa:.2f} Max size of: {max_size_qa}  Std: {std_dev_qa:.2f}")
        print(f"answers' size: {len(ans_sizes)} mean: {statistics.mean(ans_sizes):.2f} Max size of: {max(ans_sizes)}  Std: {statistics.stdev(ans_sizes):.2f}")
        print("\n")
