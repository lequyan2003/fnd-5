import os
import json
from tqdm import tqdm
import time
import numpy as np
import sys
sys.path.append('../emotion')
import extract_emotion_ch
import extract_emotion_en

# Define directories
save_dir = './data'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

datasets_ch = ['Weibo-16', 'Weibo-16-original', 'Weibo-20', 'Weibo-20-temporal']
datasets_en = ['RumourEval-19']

for dataset in datasets_ch + datasets_en:
    print('\n\n{} [{}]\tProcessing the dataset: {} {}\n'.format(
        '-'*20, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), dataset, '-'*20))

    # Select extraction package based on dataset
    if dataset in datasets_ch:
        extract_pkg = extract_emotion_ch
    else:
        extract_pkg = extract_emotion_en

    data_dir = os.path.join('../../dataset', dataset)
    output_dir = os.path.join(save_dir, dataset)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    emotion_dir = os.path.join(output_dir, 'emotions')
    if not os.path.exists(emotion_dir):
        os.mkdir(emotion_dir)

    # Handle possible issues with JSON files
    split_datasets = {}
    for split in ['train', 'val', 'test']:
        json_file_path = os.path.join(data_dir, '{}.json'.format(split))
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    split_datasets[split] = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading JSON file {json_file_path}: {e}")
                split_datasets[split] = []  # Initialize with empty list if there's an error
        else:
            print(f"JSON file {json_file_path} does not exist.")
            split_datasets[split] = []

    # Process datasets
    for t, pieces in split_datasets.items():
        if not pieces:
            print(f"No data to process for {t} in dataset {dataset}.")
            continue
        
        arr_is_saved = False
        json_is_saved = False

        # Check if the .npy and .json files are saved
        for f in os.listdir(output_dir):
            if '.npy' in f and t in f:
                arr_is_saved = True
            if t in f:
                json_is_saved = True

        if arr_is_saved:
            print(f"Array for {t} already exists. Skipping processing.")
            continue

        # If JSON is saved, load it
        if json_is_saved:
            try:
                with open(os.path.join(output_dir, '{}.json'.format(t)), 'r', encoding='utf-8') as f:
                    pieces = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading JSON file {os.path.join(output_dir, '{}.json'.format(t))}: {e}")
                pieces = []  # Initialize with empty list if there's an error
                continue  # Skip processing for this file

        # Words cutting
        if 'content_words' not in pieces[0].keys() and pieces:
            print('[{}]\tWords Cutting...'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
            for p in tqdm(pieces):
                p['content_words'] = extract_pkg.cut_words_from_text(p['content'])
                p['comments_words'] = [extract_pkg.cut_words_from_text(com) for com in p['comments']]
            try:
                with open(os.path.join(output_dir, '{}.json'.format(t)), 'w', encoding='utf-8') as f:
                    json.dump(pieces, f, indent=4, ensure_ascii=False)
            except (IOError) as e:
                print(f"Error saving JSON file {os.path.join(output_dir, '{}.json'.format(t))}: {e}")

        # Extract emotions and save
        try:
            emotion_arr = [extract_pkg.extract_dual_emotion(p) for p in tqdm(pieces)]
            emotion_arr = np.array(emotion_arr)
            print('{} dataset: got a {} emotion arr'.format(t, emotion_arr.shape))
            np.save(os.path.join(emotion_dir, '{}_{}.npy'.format(t, emotion_arr.shape)), emotion_arr)
        except Exception as e:
            print(f"Error processing emotions for {t}: {e}")
