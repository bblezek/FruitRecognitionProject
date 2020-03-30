import os, fnmatch

file_path = 'D:\\Brin\\fruit-recognition\\Plum\\'

file_list = fnmatch.filter(os.listdir(file_path), '*.png')

new_name = 'Plum'

for i, f in enumerate(file_list):
    nf = f"{new_name}{i}.png"
    os.rename(os.path.join(file_path, f), os.path.join(file_path, nf))