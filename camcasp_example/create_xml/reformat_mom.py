#!/usr/bin/env python
import sys
import numpy as np

# from https://app.ph.qmul.ac.uk/wiki/ajm:camcasp:multipoles
moments = """
   Q00
   Q10  Q11c  Q11s
   Q20  Q21c  Q21s  Q22c  Q22s
   Q30  Q31c  Q31s  Q32c  Q32s  Q33c  Q33s"""

moments = moments.split()

iread = 0
with open(sys.argv[1], 'r') as f:
    for line in f:
        words = line.split()
        if iread == 0:
            print(line, end='')
            if 'Rank' in line:
                iread = 1
                moms = []
            continue
        if iread == 1 and len(words) == 0:
            for i in range(9):
                print('%10s = %s'%(moments[i], moms[i]))
            iread = 0
            print(line, end='')
            continue
        else:
            moms.extend(words)
