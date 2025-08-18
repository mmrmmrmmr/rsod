from ultralytics import YOLO
import sys
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import datetime
import yaml
from t import login_ssh_password
def run(name):
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # name = 'vis_yolov8s_baba_4head_mid4_2_ffmmyhead4'
    # name = 'uav_yolov8s_baba_mychoose_behind_2_256256256_ffmmyhead_lossabc1'
    # name = 'yolov8s'
    # name = 'uav_yolov8s_scam_my3bone2l_3bicross4neck256256256_4fem_my3head'
    model_path_yaml = '/home/mamingrui/sod/ultralytics-main/my_model/'+name+'.yaml'
    out_dirs_path = '/home/mamingrui/sod/ultralytics-main/out_dirs/'+name

    with open(model_path_yaml, 'r', encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)

    try:
        dir_list = sorted(os.listdir(out_dirs_path), reverse=True)
        for i in dir_list:
            path = os.path.join(out_dirs_path, i, 'weights')
            x = os.listdir(path)
            if x != []:
                # model_path_yaml = os.path.join(path, 'last.pt')
                break
    except:
        pass

    model = YOLO(model=model_path_yaml, task='detect')

    # deter = yaml_data[]

    def train(model, b=yaml_data['batch']):
        return model.train(data=yaml_data['data'], 
                            epochs=1, 
                            imgsz=640, 
                            batch=b,
                            # batch=32,
                            resume=True,
                            name=now,
                            project=out_dirs_path,
                            close_mosaic=10,
                            deterministic=True,
                            cos_lr=True)
    import torch

    try:    
        result = train(model)
    except:
        torch.cuda.empty_cache()
        dir_list = sorted(os.listdir(out_dirs_path), reverse=True)
        for i in dir_list:
            path = os.path.join(out_dirs_path, i, 'weights')
            x = os.listdir(path)
            if x != []:
                model_path_yaml = os.path.join(path, 'last.pt')
                break
        model = YOLO(model=model_path_yaml, task='detect')
        result = train(model, int(yaml_data['batch']/2))
    with open(out_dirs_path+'/best.txt', 'w') as f:
        f.write(str(result.mean_results()[0])+'\n')
        f.write(str(result.mean_results()[1])+'\n')
        f.write(str(result.mean_results()[2])+'\n')
        f.write(str(result.mean_results()[3]))

    # x = torch.rand(1,3,640,640)
    # xx = model.model(x)
    # for i in range(len(xx)):
    #     print(xx[i].size())

import time
import os

if __name__ == "__main__":
    name = sys.argv[1]
    
    import paramiko
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

    def adddir(inp):
        x1 = 'rsync -av --exclude-from /home/mamingrui/sod/ultralytics-main/exclude_file.txt \
        mamingrui@10.68.52.188:/home/mamingrui/sod/ultralytics-main/'
        x2 = '/ /home/mamingrui/sod/ultralytics-main/'
        return x1 + inp + x2 + inp

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
    
    try:
        run(name)
    except:
        pass
    
    ssh_time = creatSShConnectOb('10.68.155.95', 22, 'mamingrui', 'mmr7821976431')
    chanelSSHOb_time = ssh_time.invoke_shell()
    cmd = adddir('out_dirs/'+name)
    chanel_exe_cmd(chanelSSHOb_time, cmd)
    cmd = "7821976431\n"
    chanel_exe_cmd(chanelSSHOb_time, cmd)
    time.sleep(300)
    