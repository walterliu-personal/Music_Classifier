# Arranges files internally
# By Walter Liu

import os
from random import random
import shutil

base_dir = "C:/Projects/Music_Classifier_Generator/fakes"
new_dir = "C:/Projects/Music_Classifier_Generator/dataset"


count = 10000
for img in os.listdir(base_dir):
    print(count)
    r = random()
    if 0 < r < 0.7:
        shutil.move(base_dir + "/" + img, new_dir + f"/train/not_music/fake{count}.png")
        count += 1
    elif 0.7 < r < 0.9:
        shutil.move(base_dir + "/" + img, new_dir + f"/valid/not_music/fake{count}.png")
        count += 1
    elif 0.9 < r < 1:
        shutil.move(base_dir + "/" + img, new_dir + f"/test/not_music/fake{count}.png")
        count += 1

    
