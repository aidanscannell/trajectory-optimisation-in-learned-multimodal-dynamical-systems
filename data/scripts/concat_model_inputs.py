import os
from glob import glob

import numpy as np

files = []
start_dir = os.getcwd()
pattern = "model_inputs.npz"

for dir, _, _ in os.walk(start_dir):
    files.extend(glob(os.path.join(dir, pattern)))

print(files)
for file in files:
    print('as')
    data = np.load(file)
    if 'X' in locals():
        print(X.shape)
        print(data['x'].shape)
        X = np.vstack([X, data['x']])
        Y = np.vstack([Y, data['y']])
    else:
        print('hrere')
        X = data['x']
        Y = data['y']
        # cwd = os.getcwd()

np.savez('model_inputs_combind.npz', x=X, y=Y)
# # folder_name = '../csv/26nov/1'
# folder_name = cwd
# print(folder_name)
# cwd_split = re.split('/csv/', cwd)
# print(cwd)
# model_inputs_filename = cwd_split[0] + "/npz/" + cwd_split[
#     1] + "/model_inputs.npz"
