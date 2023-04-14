# Clone or update the yolov7 repo
import os, subprocess, shutil
os.chdir('tabledetect')

if os.path.exists('yolov7'):
    # Sync
    pass
else:   
    # Clone
    command = 'git clone https://github.com/WongKinYiu/yolov7.git yolov7'; subprocess.call(command, shell=True)

