# Generates simple random melodies
# By Walter Liu

import random
import music21
import numpy as np
import time
from score_to_img import main
from music21.note import Note as n
from music21.duration import Duration as d
from copy import deepcopy as cp

NUM_IMGS = 1
start_from = 0
durs = [1/4, 1/2, 1/2, 1, 1, 1, 2]

nc41 = n("C4", duration=d(1))
ng41 = n("G4", duration=d(1))
na41 = n("A4", duration=d(1))
ng42 = n("G4", duration=d(2))
nf41 = n("F4", duration=d(1))
ne41 = n("E4", duration=d(1))
nd41 = n("D4", duration=d(1))


ns = [cp(nc41), cp(ng41), cp(na41), cp(ng42), cp(nf41), cp(ne41), cp(nd41)]

for i in range(start_from, NUM_IMGS):
    print(i)
    stre = music21.stream.Stream()
    length = random.randint(25,65)
    for l in range(length):
        norc = random.randint(0, 10)
        if norc < 9:
            randnote = cp(random.choice(ns))
            randdur = music21.duration.Duration(random.choice(durs))
            randnote.duration = randdur
            stre.append(randnote)
        elif norc >= 10:
            chordlen = random.randint(2, 3)
            ntes = []
            for n in range(chordlen):
                ntes.append(random.choice(ns))
            randchord = music21.chord.Chord(ntes)
            randdur = music21.duration.Duration(random.choice(durs))
            randchord.duration = randdur
            stre.append(randchord)
    main(stre, f"C:/users/walte/desktop/faketwinkle.png")
    print("done", i)

