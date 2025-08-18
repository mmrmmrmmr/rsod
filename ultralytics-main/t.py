import pexpect
def login_ssh_password(xx,passwd):
    if xx and passwd:
        print(xx)
        ssh = pexpect.spawn(xx)
        i = ssh.expect(['password:'], timeout=5)
        if i == 0:
            print('transporting')
            ssh.sendline(passwd)
        else:
            print('running')


# if __name__ == "__main__":
#     login_ssh_password('','mmr7821976431')
