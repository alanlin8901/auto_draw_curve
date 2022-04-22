import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
filepaths = glob.glob('./result/*.csv')
total_model = {}
for filename in filepaths:
    with open(os.path.join(os.getcwd(), filename), 'r') as f:
        model_name = os.path.split(filename)[1].split('_')[0]
        total_model[model_name] = pd.read_csv(
            f, index_col=0, squeeze=True).to_dict()

model_dicts = {}

for model_name in total_model:
    for label_name in total_model[model_name]:
        if not model_dicts.get(label_name):
            model_dicts[label_name] = {}
        model_dicts[label_name][model_name] = []
        for index in total_model[model_name][label_name]:
            value = total_model[model_name][label_name][index]
            model_dicts[label_name][model_name].append(value)

epochs = np.linspace(0, 120, 120)

color = {
    'CNN': 'r',
    'GRU': 'g',
    'Bi-LSTM' : 'b',
    'VGG16': 'o'
}
for label in model_dicts:
    for model_name in model_dicts[label]:
        value = model_dicts[label][model_name]
        plt.plot(epochs, value, c=color[model_name], label=model_name)
    file_name = f'total_{label}'
    plt.title(file_name)
    plt.legend()
    plt.savefig(f'{file_name}.png')
    plt.close()
    print(f'{label}_done')