import os, os.path
import glob
import traceback
import csv

cats = ['cpp', 'java', 'python']

c = 0
with open(os.path.join('/Users/wj/项目/ipython_root/data/lang_codes', 'train.csv'), 'w') as fptrain, \
                open(os.path.join('/Users/wj/项目/ipython_root/data/lang_codes', 'test.csv'), 'w') as fptest:
    writertrain = csv.writer(fptrain)
    writertest = csv.writer(fptest)
    for i in glob.glob('/Users/wj/项目/ipython_root/data/lang_codes/*'):
        if os.path.isdir(i):
            cat = os.path.basename(i)
            id = cats.index(cat) + 1
            for f in glob.glob(f'{i}/*'):
                try:
                    line = open(f, encoding='utf8').read().encode('unicode_escape').decode('utf-8')
                    if c % 3 == 2:
                        writertest.writerow([id, line])
                    else:
                        writertrain.writerow([id, line])
                except:
                    pass
                c += 1
            print(i)
