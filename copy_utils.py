# 文件复制
import os

src_path = r'./a1/'
target_path = r'./a2/'


# 封装成函数
def copy_function(src, target):
    filelist = os.listdir(src)
    for file in filelist:
        path = os.path.join(src, file)
        with open(path, 'rb') as read_stream:
            container = read_stream.read()
            path1 = os.path.join(target, file)
        with open(path1, 'wb') as write_stream:
            write_stream.write(container)

def get_pair(src1,src2,note):
    peos = os.listdir(src1)
    cloths = os.listdir(src2)
    with open(note,"w") as write_stream:
        for p in peos:
            for c in cloths:
                write_stream.write(" ".join([p,c])+"\n")


if __name__ == "__main__":
    src1 = r'./a1/1/'
    tar1 = r'./a2/1/'
    src2 = r'./a1/2/'
    tar2 = r'./a2/2/'
    note = r"./a0/pair.txt"
    copy_function(src1,tar1)
    copy_function(src2,tar2)
    get_pair(src1,src2,note)
