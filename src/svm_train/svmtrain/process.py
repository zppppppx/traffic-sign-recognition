import os

dir = r'.\temp'
files = os.listdir(dir)
print(files)

cnt = 0
for i in files:
    src_path = os.path.join(dir, i)
    print(src_path)
    filename = '80__' + str(cnt) + '.png'
    dst_path = os.path.join(dir, filename)
    
    cnt += 1
    os.rename(src_path, dst_path)