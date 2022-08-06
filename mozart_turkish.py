# Generates a small section of Mozart's Turkish March
# By Walter Liu

import music21 as m
from music21.note import Note as n
from music21.chord import Chord as c
from music21.note import Rest as r
from music21.duration import Duration as d
from copy import deepcopy as cp
from music21.tie import Tie as t
from score_to_img import main
import random

S = "start"
C = "continue"
X = "stop"

rh = m.stream.Part()
ts = m.meter.TimeSignature("2/4")
ks = m.key.KeySignature(0)
tmp = m.tempo.MetronomeMark(number=150)
cl = m.clef.TrebleClef()
rh.append(ts)
rh.append(tmp)
rh.append(ks)
rh.append(cl)

lh = m.stream.Part()
ts = m.meter.TimeSignature("2/4")
ks = m.key.KeySignature(0)
tmp = m.tempo.MetronomeMark(number=120)
cl = m.clef.BassClef()
lh.append(ts)
lh.append(tmp)
lh.append(ks)
lh.append(cl)

nb4025 = n("B4", duration=d(1/4))
na4025 = n("A4", duration=d(1/4))
nab4025 = n("A-4", duration=d(1/4))
nc505 = n("C5", duration=d(1/2))
r05 = r(duration=d(1/2))
nd5025 = n("D5", duration=d(1/4))
nc5025 = n("C5", duration=d(1/4))
ne505 = n("E5", duration=d(1/2))
nf5025 = n("F5", duration=d(1/4))
neb5025 = n("E-5", duration=d(1/4))
nb5025 = n("B5", duration=d(1/4))
na5025 = n("A5", duration=d(1/4))
nab5025 = n("A-5", duration=d(1/4))
nc61 = n("C6", duration=d(1))
na505 = n("A5", duration=d(1/2))
nc605 = n("C6", duration=d(1/2))
ng50125 = n("G5", duration=d(1/8))
na50125 = n("A5", duration=d(1/8))
ngb505 = n("G-5", duration=d(1/2))
ng505 = n("G5", duration=d(1/2))
neb505 = n("E-5", duration=d(1/2))
ne51 = n("E5", duration=d(1))
ne5025 = n("E5", duration=d(1/4))

na305 = n("A3", duration=d(1/2))
nc405 = n("C4", duration=d(1/2))
ne405 = n("E4", duration=d(1/2))
ne305 = n("E3", duration=d(1/2))
nb305 = n("B3", duration=d(1/2))
nb205 = n("B2", duration=d(1/2))
ne31 = n("E3", duration=d(1))

cgb5a505 = c([ngb505, na505], duration=d(1/2))
ce5g505 = c([ne505, ng505], duration=d(1/2))
ceb5gb505 = c([neb505, ngb505], duration=d(1/2))

cc4e405 = c([nc405, ne405], duration=d(1/2))
ce3b305 = c([ne305, nb305], duration=d(1/2))

r1 = r(duration=d(1))
r2 = r(duration=d(2))

rh.append(r1)
lh.append(r2)
rh.append(cp(nb4025))
rh.append(cp(na4025))
rh.append(cp(nab4025))
rh.append(cp(na4025))
rh.append(cp(nc505))
rh.append(cp(r05))
rh.append(cp(nd5025))
rh.append(cp(nc5025))
rh.append(cp(nb4025))
rh.append(cp(nc5025))
rh.append(cp(ne505))
rh.append(cp(r05))
rh.append(cp(nf5025))
rh.append(cp(ne5025))
rh.append(cp(neb5025))
rh.append(cp(ne5025))
rh.append(cp(nb5025))
rh.append(cp(na5025))
rh.append(cp(nab5025))
rh.append(cp(na5025))
rh.append(cp(nb5025))
rh.append(cp(na5025))
rh.append(cp(nab5025))
rh.append(cp(na5025))
rh.append(cp(nc61))
rh.append(cp(na505))
rh.append(cp(nc605))
rh.append(cp(ng50125))
rh.append(cp(na50125))
rh.append(cp(nb5025))
rh.append(cp(cgb5a505))
rh.append(cp(ce5g505))
rh.append(cp(cgb5a505))
rh.append(cp(ng50125))
rh.append(cp(na50125))
rh.append(cp(nb5025))
rh.append(cp(cgb5a505))
rh.append(cp(ce5g505))
rh.append(cp(cgb5a505))
rh.append(cp(ng50125))
rh.append(cp(na50125))
rh.append(cp(nb5025))
rh.append(cp(cgb5a505))
rh.append(cp(ce5g505))
rh.append(cp(ceb5gb505))
rh.append(cp(ne51))

lh.append(cp(na305))
lh.append(cp(cc4e405))
lh.append(cp(cc4e405))
lh.append(cp(cc4e405))
lh.append(cp(na305))
lh.append(cp(cc4e405))
lh.append(cp(cc4e405))
lh.append(cp(cc4e405))
lh.append(cp(na305))
lh.append(cp(cc4e405))
lh.append(cp(na305))
lh.append(cp(cc4e405))
lh.append(cp(na305))
lh.append(cp(cc4e405))
lh.append(cp(cc4e405))
lh.append(cp(cc4e405))
lh.append(cp(ne305))
lh.append(cp(ce3b305))
lh.append(cp(ce3b305))
lh.append(cp(ce3b305))
lh.append(cp(ne305))
lh.append(cp(ce3b305))
lh.append(cp(ce3b305))
lh.append(cp(ce3b305))
lh.append(cp(ne305))
lh.append(cp(ce3b305))
lh.append(cp(nb205))
lh.append(cp(nb305))
lh.append(cp(ne31))

tm = m.stream.Score()
tm.append(rh)
tm.append(lh)

tm.parts[0].offset = 0
tm.parts[1].offset = 0

tm = tm.chordify()

tm.show()

main(tm, "C:/Users/walte/desktop/mozart_turkish.png")