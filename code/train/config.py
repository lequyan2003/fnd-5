# All the datasets (Chinese and English)
datasets_ch = ['Weibo-16', 'Weibo-16-original', 'Weibo-20', 'Weibo-20-temporal']
datasets_en = ['RumourEval-19']

# All the models to be tested
model_names = ['MLP', 'BiGRU', 'EmotionEnhancedBiGRU']

# Experiment settings
experimental_dataset = datasets_ch[0]  # Default dataset for experiments
experimental_model_name = model_names[2]  # Default model for experiments

epochs = 50  # Number of epochs for training
batch_size = 32  # Size of each training batch

l2_param = 0.01  # L2 regularization parameter
lr_param = 0.001  # Learning rate
