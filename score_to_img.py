# Converts a music21.stream.Score object to a .png image
# By Walter Liu

import music21
import cv2
from music21.stream.base import Measure
import numpy as np
import time



def class_only(stre, theclass):
    res = []
    for i in range(len(stre)):
        if type(stre[i]) == theclass:
            res.append(stre[i])
    return res

def rectify(stre):

    tsn = stre.recurse().getElementsByClass(music21.meter.TimeSignature)[0]
    ts = tsn.numerator*(4/tsn.denominator)
    mess = class_only(stre, music21.stream.Measure)
    for i in range(len(mess)):
        if i != 0 and type(mess[i]) == music21.stream.Measure:
            if type(mess[i-1]) == music21.stream.Measure:
                if mess[i].offset - mess[i-1].offset != ts:
                    mess[i].offset = mess[i-1].offset + ts

    return mess

def main(test_score, outpath):
    tchords = test_score.chordify()
    #tchords = rectify(tchords)
    allchords = []
    maxm = -1
    for m in tchords:
        allchords.append(m)
        if m.offset > maxm:
            maxm = m.offset
        continue
        #if type(m) == music21.stream.Measure:
        #    mess = class_only(m, music21.chord.Chord)
        #    for n in mess:
        #        n.offset += m.offset
        #        if n.offset > maxm:
        #            maxm = n.offset
        #        allchords.append(n)
    
    # Lower if model training too long
   # print(allchords)
    width = 4096
    newimg = np.zeros((128,width, 1), np.uint8)
    for chor in allchords:
        if type(chor) == music21.chord.Chord:
            for n in chor:
                newimg[n.pitch.midi, int(chor.offset * (width / maxm)) : int((chor.offset + n.duration.quarterLength) * (width / maxm))] = 255
    
    
    
    #cv2.imshow("pls", newimg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    cv2.imwrite(outpath, newimg)
    