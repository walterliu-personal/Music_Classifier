# Combs the music21 corpus for scores and transforms them into images for training
# By Walter Liu

import music21
import os
from score_to_img import main

folders = os.listdir(r"c:\users\walte\miniconda3\Lib\site-packages\music21\corpus")

folders.remove("__pycache__")
folders.remove("_metadataCache")
folders.remove("__init__.py")
for item in folders:
    if item.count("."):
        folders.remove(item)

folders = folders[folders.index("cpebach"):]

print(folders)

allsongs = []
for item in folders:
    allsongs += music21.corpus.getComposer(item)

print(len(allsongs))

count = 0
for s in allsongs:
    print(s)
    s = music21.corpus.parse(s)
    if type(s) == music21.stream.Opus:
        for score in s:
            try:
                main(score, f"c:\projects\music_classifier_generator\imgs\img{count}.png")
                count += 1
            except:
                continue
    else:
        try:
            main(s, f"c:\projects\music_classifier_generator\imgs\img{count}.png")
            count += 1
        except:
            continue
    print(count)


    