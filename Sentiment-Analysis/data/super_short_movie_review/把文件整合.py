import glob
import os
path = 'data_deal'
dirs = os.listdir(path)
for dir in dirs:
    subpath = path+ '\\' + dir
    subdirs = os.listdir(subpath)
    for subdir in subdirs:
        f = open(dir + '_' + subdir +'.txt', 'wb')
        pathway = path + '\\'+dir+ '\\' + subdir + '\\*.txt'
        files = glob.glob(pathway)
        for filename in files:
            txt_file = open(filename, 'r')
            buf = txt_file.read()
            buf = buf.replace('\n','')
            f.write(buf+'\n')
        f.close()
