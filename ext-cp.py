import argparse

parser = argparse.ArgumentParser(description='复制指定扩展名的文件到指定文件夹.')
parser.add_argument('--srcdir', required=True, help='源文件夹')
parser.add_argument('--ext', required=True, help='文件扩展名')
parser.add_argument('--dstdir', required=True, help='目标文件夹')

args = parser.parse_args()
print(args)
srcdir = args.srcdir
exts = args.ext.split(',')
dstdir = args.dstdir

# srcdir = '/Users/wj/项目/ws_10086'
# ext = 'java'
# dstdir = '/Users/wj/项目/ipython_root/data/lang_codes/java'

import os
import os.path
import shutil
import traceback


def build_file_name(dir, file_name):
    count = 1
    name = file_name
    while True:
        path = os.path.join(dir, name)
        if not os.path.exists(path):
            return path
        name = f'{file_name}-{count}'
        count += 1
        if count > 10:
            break


exts = [f'.{ext}' for ext in exts]
for root, dirs, files in os.walk(srcdir):
    # for dir in dirs:
    #     print(os.path.join(root, dir))
    for file in files:
        srcfile = os.path.join(root, file)
        dstfile = build_file_name(dstdir, file)
        if not dstfile:
            continue
        if os.path.splitext(file)[-1] in exts:
            try:
                shutil.copy(srcfile, dstfile)
                print(srcfile)
            except:
                traceback.print_exc()

