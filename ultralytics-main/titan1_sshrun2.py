import time
import paramiko

name = 'vis_yolov8s'

def chanel_exe_cmd(ChanelSSHOb, cmd, t=3):
    ChanelSSHOb.send(cmd)
    ChanelSSHOb.send("\n")
    time.sleep(t)
    resp = ChanelSSHOb.recv(9999).decode("utf8")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("Exec sshCmd: %s" % (cmd))
    print("--------------------")
    print("Exec Result: %s" % (resp))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    return resp


def creatSShConnectOb(ip_remote, port_remote, username, password):
    print('---------- start to create SSH object')
    print('Remote SSH Info:\n\'ip:%s  port:%d  username:%s  password:%s\'\n' % (
        ip_remote, port_remote, username, password))
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(ip_remote, port_remote, username=username, password=password, timeout=60)  # timeout protection
        return ssh
    except:
        print('Warning:\nFist connect the ABC failed, now will retry!')
        ssh.connect(ip_remote, port_remote, username=username, password=password, timeout=60)  # timeout re-try
        print('Error:\nAttempt to connect ABC failed!!! Please check the IP / port/ account / password.')

input1 = [
'rsync -av --exclude-from /home/mamingrui/sod/ultralytics-main/exclude_file.txt mamingrui@10.68.155.95:/home/mamingrui/sod/ultralytics-main/my_model/ /home/mamingrui/sod/ultralytics-main/my_model',
'rsync -av --exclude-from /home/mamingrui/sod/ultralytics-main/exclude_file.txt mamingrui@10.68.155.95:/home/mamingrui/sod/ultralytics-main/ultralytics/ /home/mamingrui/sod/ultralytics-main/ultralytics',
# 'rsync -av --exclude-from /home/mamingrui/sod/ultralytics-main/exclude_file.txt mamingrui@10.68.155.95:/home/mamingrui/sod/ultralytics-main/titan1_sshrun.py /home/mamingrui/sod/ultralytics-main/titan1_sshrun.py'
'rsync -av --exclude-from /home/mamingrui/sod/ultralytics-main/exclude_file.txt mamingrui@10.68.155.95:/home/mamingrui/sod/ultralytics-main/traint_2.py /home/mamingrui/sod/ultralytics-main/traint_2.py'
]

def adddir(inp):
    x1 = 'rsync -av --exclude-from /home/mamingrui/sod/ultralytics-main/exclude_file.txt \
    /home/mamingrui/sod/ultralytics-main/'
    x2 = '/ mamingrui@10.68.155.95:/home/mamingrui/sod/ultralytics-main/'
    return x1 + inp + x2 + inp

def timecuda():
    pass

def run():
    ssh = creatSShConnectOb('10.68.52.188', 22, 'mamingrui', '7821976431')
    chanelSSHOb = ssh.invoke_shell()  # 建立交互式的shell
    chanel_exe_cmd(chanelSSHOb, '\n')
    time.sleep(5)
    for i in input1:
        print(i)
        chanel_exe_cmd(chanelSSHOb, i)
        cmd = "mmr7821976431\n"
        chanel_exe_cmd(chanelSSHOb, cmd)
    cmd = 'nohup /home/mamingrui/.conda/envs/frame/bin/python /home/mamingrui/sod/ultralytics-main/traint_2.py ' + name +' > out2.out'
    chanel_exe_cmd(chanelSSHOb, cmd)
    # time.sleep(30)

    ssh_time = creatSShConnectOb('10.68.52.188', 22, 'mamingrui', '7821976431')
    chanelSSHOb_time = ssh_time.invoke_shell()

    cmd = adddir('out_dirs/'+name)
    chanel_exe_cmd(chanelSSHOb_time, cmd)
    cmd = "mmr7821976431\n"
    chanel_exe_cmd(chanelSSHOb_time, cmd)

    time.sleep(300)

if __name__ == '__main__':
    run()