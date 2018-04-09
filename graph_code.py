output_directory= '/Users/tala/documents/tensorflow/script/output'
import glob
import collections
import os
import matplotlib.pyplot as plt
import pandas as pd

def GetFlagsDict(filename):
    elements = filename.split('/')[-1][:-4].split('_')
    return dict([s.split('-') for s in elements])


groups = collections.defaultdict(list)
for filename in glob.glob(os.path.join(output_directory, "*.csv")):
    d = GetFlagsDict(filename)
    strategy = d.pop('s')
    group_key = '_'.join(['%s-%s' % (k, v)  for k, v in sorted(d.items())])
    groups[group_key].append(filename)

print(groups)

for group_key, files in groups.items():
    fig = plt.figure()
    fig, axis = plt.subplots(figsize=(15,7))

    for f in files:
        data = pd.read_csv(f)
        d = GetFlagsDict(f)
        axis.plot(data['Number of examples'], data['Accuracy Score'], label=d['s'])
        plt.legend()
    plt.title(group_key)
    plt.show()
    fig.savefig(os.path.join(output_directory, group_key + '.png'))
