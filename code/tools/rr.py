import sys, warnings, os
warnings.filterwarnings("ignore")
import paramiko, logging
logging.getLogger("paramiko").setLevel(logging.CRITICAL)

HOST, PORT, USER, PW = "610854wm2fi2.vicp.fun", 41484, "yaokang", "123456"


def client():
    import time
    last = None
    for i in range(25):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30, banner_timeout=60, auth_timeout=30)
            return c
        except Exception as e:
            last = e
            time.sleep(min(30, 4 + 3 * i))
    raise last


def run(cmd, timeout=600):
    c = client()
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout, get_pty=False)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    c.close()
    return code, out, err


def put(local, remote):
    import base64, shlex
    data = open(local, "rb").read()
    c = client()
    rq = shlex.quote(remote)
    stdin, stdout, stderr = c.exec_command(f"mkdir -p $(dirname {rq}) && base64 -d > {rq}.tmp && mv {rq}.tmp {rq}")
    stdin.write(base64.b64encode(data))
    stdin.channel.shutdown_write()
    code = stdout.channel.recv_exit_status()
    err = stderr.read().decode()
    c.close()
    if code != 0:
        raise RuntimeError(err)


def get(remote, local):
    c = client()
    s = c.open_sftp()
    s.get(remote, local)
    s.close(); c.close()


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "run":
        cmd = sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read()
        t = int(os.environ.get("RR_TIMEOUT", "600"))
        code, out, err = run(cmd, t)
        sys.stdout.buffer.write(out.encode("utf-8", "replace"))
        if err.strip():
            sys.stdout.buffer.write(("\n[stderr]\n" + err[-6000:]).encode("utf-8", "replace"))
        sys.stdout.buffer.write(f"\n[exit {code}]\n".encode())
    elif mode == "put":
        put(sys.argv[2], sys.argv[3]); print("ok")
    elif mode == "get":
        get(sys.argv[2], sys.argv[3]); print("ok")
