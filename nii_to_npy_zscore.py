import numpy as np
import nibabel as nib
import os
import scipy.stats as stats


source_dir = "new_data_prob"
dest_dir = "new_data_zscore"

if not os.path.exists(dest_dir):
  os.mkdir(dest_dir)

for filename in os.listdir(source_dir):
    f = os.path.join(source_dir, filename)
    # checking if it is a file
    if os.path.isfile(f):
        img = nib.load(f)
        a = np.array(img.dataobj)
        newmat = stats.zscore(a, axis=None)
        #print(new_fn)
        new_fn = filename[:-7]
        np.save(os.path.join(dest_dir, new_fn), newmat)