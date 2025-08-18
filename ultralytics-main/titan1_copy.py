from t import login_ssh_password

name = 'vis_yolov8s'

x1 = 'rsync -av --exclude-from /home/mamingrui/sod/ultralytics-main/exclude_file.txt \
mamingrui@10.68.155.95:/home/mamingrui/sod/ultralytics-main/'
x2 = '/ /home/mamingrui/sod/ultralytics-main/'
x3 = ' /home/mamingrui/sod/ultralytics-main/'
def adddir(inp):
    return x1 + inp + x2 + inp

def addx(inp):
    return x1 + inp + x3 + inp
print(addx('t.py'))
print(adddir('my_model'))
print(adddir('ultralytics'))
print(addx('traint_1.py'))

from titan1_sshrun import creatSShConnectOb, adddir, chanel_exe_cmd
def run():
    ssh = creatSShConnectOb('10.68.52.188', 22, 'mamingrui', '7821976431')
    chanelSSHOb = ssh.invoke_shell()  # 建立交互式的shell


    cmd = adddir('out_dirs/'+name)
    chanel_exe_cmd(chanelSSHOb, cmd)
    cmd = "mmr7821976431\n"
    chanel_exe_cmd(chanelSSHOb, cmd)




