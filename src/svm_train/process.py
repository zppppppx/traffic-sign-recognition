import os

root_path = './raw_data'
files = os.listdir(root_path)

i = 0
for fig in files:
    cls = int(fig[:3])
    if cls < 8:
        fig_path = os.path.join(root_path, fig)
        new_name = os.path.join(root_path, '000_'+str(i)+'.png')
        os.rename(fig_path, new_name)
        i += 1

