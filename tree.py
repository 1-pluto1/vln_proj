import pathlib, sys

def print_tree(path='.', max_depth=5, ignore=None):
    ignore = ignore or {'__pycache__', '.git', '.idea', 'build', 'dist'}
    
    def _tree(p, depth, prefix):
        if depth > max_depth or p.name in ignore:
            return
        print(prefix + p.name + ('/' if p.is_dir() else ''))
        if p.is_dir():
            for i, child in enumerate(sorted(p.iterdir())):
                is_last = i == len(list(p.iterdir())) - 1
                _tree(child, depth + 1, prefix + ('    ' if is_last else '│   '))
    
    _tree(pathlib.Path(path), 0, '')

print_tree('/home/gentoo/docker_shared/asus/liusq/UAV_VLN/vln_proj/TravelUAV')