from config import model_names
from sklearn.metrics import accuracy_score, classification_report
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import numpy as np
import os
import json
import math
import sys
sys.path.append('../model')
from MLP import MLP5Layers
from BiGRU import EmotionEnhancedBiGRU


labels_names = ['fake', 'real', 'unverified']
datasets_ch = ['Weibo-16', 'Weibo-16-original',
               'Weibo-20', 'Weibo-20-temporal']
datasets_en = ['RumourEval-19']


dataset_dir = '../preprocess/data'
results_dir = './results'
if not os.path.exists(results_dir):
    os.mkdir(results_dir)


def calculate_RMSE_of_RumourEval(y_pred, test_label):
    """
    Reference: https://figshare.com/articles/dataset/RumourEval_2019_data/8845580, rumoureval2019/home_scorer_macro.py
    """
    errors = []
    for i in range(len(y_pred)):
        pred_label = y_pred[i].argmax()

        # unverified
        if pred_label == 2:
            yhat = 'unverified'
            confidence = .0
        else:
            yhat = 'fake' if pred_label == 0 else 'real'
            confidence = y_pred[i][pred_label]
            confidence /= np.sum(y_pred[i][:2])

        groud_truth = test_label[i].argmax()
        if pred_label == groud_truth and (yhat == "fake" or yhat == "real"):
            errors.append((1-confidence) ** 2)
        elif groud_truth == 2:
            errors.append((confidence) ** 2)
        else:
            errors.append(1.0)

    return math.sqrt(sum(errors)/len(errors))


def predict_single_output(y_pred, test_label):
    y_pred_label = np.zeros(y_pred.shape)
    for i, arg in enumerate(y_pred.argmax(axis=1)):
        y_pred_label[i][arg] = 1

    names = labels_names[:test_label.shape[-1]]

    print()
    print('TEST_sz: {}'.format(len(test_label)))
    for i, name in enumerate(names):
        arr = test_label[:, i]
        print('{}_sz: {}'.format(name, len(arr[arr == 1])))
    print()
    accuracy = accuracy_score(test_label, y_pred_label)
    print('Accuracy: {:.3f}'.format(accuracy))
    print()
    print(classification_report(test_label, y_pred_label,
                                labels=[i for i in range(len(names))],
                                target_names=names, digits=3, output_dict=False))
    report = classification_report(test_label, y_pred_label,
                                   labels=[i for i in range(len(names))],
                                   target_names=names, digits=3, output_dict=True)
    print()
    print()
    return accuracy, report


def load_dataset(dataset, input_types=['emotions']):
    assert dataset in datasets_ch + datasets_en
    assert len(input_types) >= 1
    for t in input_types:
        assert t in ['emotions', 'semantics']

    label_dir = os.path.join(dataset_dir, dataset, 'labels')
    for f in os.listdir(label_dir):
        f = os.path.join(label_dir, f)
        if 'train_' in f:
            train_label = np.load(f)
        elif 'val_' in f:
            val_label = np.load(f)
        elif 'test_' in f:
            test_label = np.load(f)

    train_data, val_data, test_data = [], [], []
    semantics_embedding_matrix = None
    for t in input_types:
        data_dir = os.path.join(dataset_dir, dataset, t)
        for f in os.listdir(data_dir):
            f = os.path.join(data_dir, f)
            if 'train_' in f:
                train_data.append(np.load(f))
            elif 'val_' in f:
                val_data.append(np.load(f))
            elif 'test_' in f:
                test_data.append(np.load(f))
            elif 'embedding_matrix_' in f:
                semantics_embedding_matrix = np.load(f)

    if len(input_types) == 1:
        train_data, val_data, test_data = train_data[0], val_data[0], test_data[0]

    data = [train_data, val_data, test_data]
    label = [train_label, val_label, test_label]

    print()
    for i, t in enumerate(['Train', 'Val', 'Test']):
        if len(input_types) == 1:
            print('{} data: {}, {} label: {}'.format(
                t, data[i].shape, t, label[i].shape))
        else:
            print('{} data:'.format(t))
            for j, it in enumerate(input_types):
                print('[{}]\t{}'.format(it, data[i][j].shape))
            print('{} label: {}\n'.format(t, label[i].shape))
    print()

    if 'semantics' in input_types:
        return data, label, semantics_embedding_matrix
    else:
        return data, label


def calculate_balanced_sample_weights(train_label):
    weights = np.ones(len(train_label))

    label_sizes = dict()
    print('\nIn train_label {}:'.format(train_label.shape))
    for i, name in enumerate(labels_names[:train_label.shape[-1]]):
        arr = train_label[:, i]
        sz = len(arr[arr == 1])
        label_sizes[name] = sz
        print('{}_sz: {}'.format(name, sz))
    print()

    min_size = min(label_sizes.values())

    for label, size in label_sizes.items():
        index = labels_names.index(label)
        weights[train_label.argmax(axis=1) == index] = size / min_size

    return weights


def train(model, dataset, data, label, model_name, epochs=50, batch_size=32, use_sample_weights=False):
    print('\n{} Train {}\n'.format('-'*20, '-'*20))

    train_data, val_data, test_data = data
    train_label, val_label, test_label = label

    results_dataset_dir = os.path.join(results_dir, dataset)
    if not os.path.exists(results_dataset_dir):
        os.mkdir(results_dataset_dir)
    results_model_file = os.path.join(
        results_dataset_dir, '{}.hdf5'.format(model_name))

    # Train
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    checkpoint = ModelCheckpoint(
        results_model_file, monitor='val_loss', save_best_only=True, save_weights_only=True)

    sample_weights = calculate_balanced_sample_weights(
        train_label) if use_sample_weights else None
    print('Sample Weights when traning: \n{}\n'.format(sample_weights))

    model.fit(train_data, train_label, epochs=epochs, batch_size=batch_size, sample_weight=sample_weights,
              validation_data=(val_data, val_label), shuffle=True, callbacks=[checkpoint, early_stop])

    # Load the best model and predict
    model.load_weights(results_model_file)

    for i, t in enumerate(['val', 'test']):
        print('\n{} {} {}\n'.format('-'*20, t, '-'*20))
        y_pred = model.predict(data[1+i])
        accuracy, report = predict_single_output(y_pred, label[1+i])

        results = {'dataset': t, 'samples': len(y_pred), 'accuracy': accuracy,
                   'classification_report': report}
        if dataset in datasets_en:
            rmse = calculate_RMSE_of_RumourEval(y_pred, label[1+i])
            results['RMSE'] = rmse

        results_file = results_model_file.replace(
            '.hdf5', '_{}.json'.format(t))
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)


def main(experimental_dataset, experimental_model_name, epochs, batch_size, l2_param, lr_param):
    # MLP, inputs are only emotions
    if experimental_model_name == model_names[0]:
        data, label = load_dataset(
            experimental_dataset, input_types=['emotions'])
        model = MLP5Layers(
            input_dim=data[0].shape[-1], category_num=label[0].shape[-1], l2_param=l2_param, lr_param=lr_param).model

    # BiGRU, inputs are only semantics
    elif experimental_model_name == model_names[1]:
        data, label, embedding_matrix = load_dataset(
            experimental_dataset, input_types=['semantics'])

        CONTENT_WORDS = 100 if experimental_dataset in datasets_ch else 50
        model = EmotionEnhancedBiGRU(max_sequence_length=CONTENT_WORDS,
                                     embedding_matrix=embedding_matrix, emotion_dim=0, category_num=label[0].shape[-1], l2_param=l2_param, lr_param=lr_param).model

    # EmotionEnhancedBiGRU, inputs are emotions and semantics
    else:
        data, label, embedding_matrix = load_dataset(
            experimental_dataset, input_types=['semantics', 'emotions'])

        CONTENT_WORDS = 100 if experimental_dataset in datasets_ch else 50
        model = EmotionEnhancedBiGRU(max_sequence_length=CONTENT_WORDS,
                                     embedding_matrix=embedding_matrix,
                                     emotion_dim=data[0][1].shape[-1],
                                     category_num=label[0].shape[-1],
                                     l2_param=l2_param, lr_param=lr_param).model

    print()
    print(model.summary())
    print()
    train(model=model, dataset=experimental_dataset, data=data,
          label=label, model_name=experimental_model_name, epochs=epochs, batch_size=batch_size)
