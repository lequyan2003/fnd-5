from train import main
from config import model_names, datasets_ch, datasets_en
from config import experimental_dataset, experimental_model_name, epochs, batch_size, l2_param, lr_param
import time
import argparse

# Argument parser setup
parser = argparse.ArgumentParser(description='Run experiments with specified dataset and model.')
parser.add_argument('--dataset', default=experimental_dataset, help='Name of the dataset')
parser.add_argument('--model', default=experimental_model_name, help='Name of the model')
args = parser.parse_args()

# Validate arguments
if args.dataset not in datasets_ch + datasets_en:
    raise ValueError(f"Dataset '{args.dataset}' is not in the list of datasets.")
if args.model not in model_names:
    raise ValueError(f"Model '{args.model}' is not in the list of models.")

# Update experimental dataset and model name based on arguments
experimental_dataset = args.dataset
experimental_model_name = args.model

# Option to run all models on all datasets or a specific configuration
run_all_models = False  # Set to True if you want to run all models

if run_all_models:
    # Run all the models on all the datasets
    for dataset in datasets_ch + datasets_en:
        for model_name in model_names:
            print('================ [{}] ================'.format(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            print('[Dataset]\t{}'.format(dataset))
            print('[Model]\t{}'.format(model_name))
            print()
            main(dataset, model_name, epochs, batch_size, l2_param, lr_param)
else:
    # Run the specific model on the specific dataset
    print('================ [{}] ================'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    print('[Dataset]\t{}'.format(experimental_dataset))
    print('[Model]\t{}'.format(experimental_model_name))
    print()
    print('The hyperparameters: ')
    print('[Epoch]\t{}'.format(epochs))
    print('[Batch Size]\t{}'.format(batch_size))
    print('[L2 param]\t{}'.format(l2_param))
    print('[Learning Rate]\t{}'.format(lr_param))
    print()

    main(experimental_dataset, experimental_model_name, epochs, batch_size, l2_param, lr_param)
