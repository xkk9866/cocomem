"""Push local code dir to server as a tar (base64 over ssh)."""
import io, os, sys, tarfile, base64
sys.path.insert(0, os.path.dirname(__file__))
import rr
LOCAL = r"Y:\xkk_workspace\workspace\ICDE\2\code"
buf = io.BytesIO()
ONLY = set(sys.argv[1:])          # optional: file names to push (default: everything)
with tarfile.open(fileobj=buf, mode="w:gz") as tf:
    for root, dirs, files in os.walk(LOCAL):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", "tools", "multiseed_pending", "pre_multiseed_backup")]
        for f in files:
            if f.endswith((".py", ".sh", ".json", ".txt")) and (not ONLY or f in ONLY):
                p = os.path.join(root, f)
                data = open(p, "rb").read()
                if f.endswith((".py", ".sh")):
                    data = data.replace(b"\r\n", b"\n")      # scripts written on Windows must run under bash
                info = tarfile.TarInfo(os.path.relpath(p, LOCAL).replace("\\", "/"))
                info.size, info.mtime, info.mode = len(data), int(os.path.getmtime(p)), 0o755
                tf.addfile(info, io.BytesIO(data))
data = buf.getvalue()
c = rr.client()
stdin, stdout, stderr = c.exec_command("mkdir -p /data/yaokang/corrmem/code && cd /data/yaokang/corrmem/code && base64 -d | tar xzf - && ls | wc -l")
stdin.write(base64.b64encode(data)); stdin.channel.shutdown_write()
print(stdout.read().decode().strip(), stderr.read().decode().strip()[-500:])
c.close()
