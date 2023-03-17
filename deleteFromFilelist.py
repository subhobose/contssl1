import os

filedata = []
basedir = '/data/datasets/kDownloader/kinetics-dataset/k700-2020/frames'
with open("/home/subhrangsu/contssl/continuous_ssl_problem/datasets/lists/kinetics_filelist.txt", 'r') as f:
    filedata = list(f)

print(len(filedata))            
for fi in filedata[:]:
    filename = fi.split(" ")[0]
    fname = filename.rpartition("/")[-1]
    flag = 0
    pDirectory = filename.rpartition("/")[0].split("/")[-1]
    for root, subdirs, files in os.walk(basedir + '/' + pDirectory):
        for d in subdirs:
            if d == fname:
                print("{} Found!".format(filename))
                flag = 1
                break
    if flag == 0:
        print("Deleting filename {}".format(filename))
        filedata.remove(fi)
    
print("Deletion Complete\n")
print(len(filedata))

with open("/home/subhrangsu/contssl/continuous_ssl_problem/datasets/lists/kinetics_filelist-final.txt", 'w') as f:
    f.writelines(filedata)


  