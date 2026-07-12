"""读取四个文件"""
import glob, os

base = r'D:\shumo\evalation modei test'
files = glob.glob(os.path.join(base, '*'))

for f in sorted(files):
    name = os.path.basename(f)
    print(f"文件: {name}")
    print(f"大小: {os.path.getsize(f)} bytes")
    print("---")
