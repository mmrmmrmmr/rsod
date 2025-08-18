import paramiko

name = 'vis_yolov8s'

class MySshClient:
    def __init__(self, ssh_client):
        self.ssh_client = ssh_client
    def exec_command(self, cmd):
        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(cmd)
            return stdin, stdout, stderr
        except Exception as e:
            print(f"Error executing command {cmd}: {e}")
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ssh_client.close()

def connect(host, port, username, password):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.WarningPolicy())
    try:
        ssh.load_system_host_keys()
        ssh.connect(host, port, username, password,timeout=3)
    except paramiko.AuthenticationException:
        raise Exception(f"在主机 {host}连接失败,请检查你的参数")
    except paramiko.SSHException as e:
        raise Exception(f"在 {host}连接出错: {e}")
    except paramiko.BadHostKeyException as e:
        raise Exception(f" {host} 无法验证通过: {e}")
    except Exception as e:
        raise Exception(f" 连接到{host}:{port}: {e}超时")
    return ssh

input1 = '/home/mamingrui/.conda/envs/frame/bin/python /home/mamingrui/sod/ultralytics-main/titan1_copy.py'
input2 = '/home/mamingrui/.conda/envs/frame/bin/python /home/mamingrui/sod/ultralytics-main/traint_1.py ' + name 
ssh = connect('10.68.52.188', '22', 'mamingrui', '7821976431')

with MySshClient(ssh) as client:
    print(input1)
    stdin, stdout, stderr = client.exec_command(input1)
    with stdout:
        print(stdout.read().decode())
    with stderr:
        print(stderr.read().decode())
    print(input2)
    stdin, stdout, stderr = client.exec_command(input2)
    with stdout:
        print(stdout.read().decode())
    with stderr:
        print(stderr.read().decode())
    

