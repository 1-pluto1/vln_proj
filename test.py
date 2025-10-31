import os, re
def in_container():
    if os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv'):
        return True
    for path in ('/proc/self/cgroup', '/proc/1/cgroup'):
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                data = f.read()
            if re.search(r'(docker|containerd|kubepods)', data):
                return True
        except Exception:
            pass
    return False

print("在 Docker/容器内" if in_container() else "不在 Docker/容器内")