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

def subprocess(mom_file):
    momlines=[]
    iread = 0
    with open(mom_file, 'r') as f:
        for line in f:
            words = line.split()
            if iread == 0:
                momlines.append(line.rstrip('\n'))
                # print(line, end='')
                if 'Rank' in line:
                    iread = 1
                    moms = []
                continue
            if iread == 1 and len(words) == 0:
                for i in range(9):
                    momlines.append('%10s = %s'%(moments[i], moms[i]))
                    # print('%10s = %s'%(moments[i], moms[i]))
                iread = 0
                momlines.append(line.rstrip('\n'))
                # print(line, end='')
                continue
            else:
                moms.extend(words)
    return momlines

def main():
    mom_file='./EC/EC/OUT/EC_ISA-GRID.mom'
    momlines=subprocess(mom_file)
    f=open('reform.mom','w')
    for line in momlines:
        print(line,file=f)

if __name__ == "__main__":
    main()
