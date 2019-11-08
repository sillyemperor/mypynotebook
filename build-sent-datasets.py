import os, os.path
import glob
import traceback
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--srcdir', required=True)
parser.add_argument('--cats', required=True, nargs='+')
parser.add_argument('--dstdir', required=True)

args = parser.parse_args()
print(args)

srcdir = args.srcdir
cats = args.cats
dstdir = args.dstdir

with open(os.path.join(dstdir, 'classes.txt'), 'w') as fp:
    for i, c in enumerate(cats, 1):
        fp.write(f'{i}:{c}')
        fp.write('\r\n')

c = 0
with open(os.path.join(dstdir, 'train.csv'), 'w') as fptrain, \
                open(os.path.join(dstdir, 'test.csv'), 'w') as fptest:
    writertrain = csv.writer(fptrain)
    writertest = csv.writer(fptest)
    for i in glob.glob(f'{srcdir}/*'):
        if os.path.isdir(i):
            cat = os.path.basename(i)
            if cat not in cats:
                continue
            id = cats.index(cat) + 1
            for f in glob.glob(f'{i}/*'):
                try:
                    line = open(f,'r').read()
                    if not line:
                        continue
                    if c % 3 == 2:
                        writertest.writerow([id, line])
                    else:
                        writertrain.writerow([id, line])
                except:
                    pass
                c += 1
            print(i)
