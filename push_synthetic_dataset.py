from datasets import load_dataset

dataset = load_dataset('json', data_files='/add-path-here')
dataset.push_to_hub("UserName/data-set-name")
