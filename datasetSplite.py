import os

import numpy as np
import pandas as pd


def main():
    g = os.walk(r'.\lfw')
    L1 = []
    for path, dir_list, file_list in g:
        # print(path[6:])
        for file_name in file_list:
            imgName = os.path.join(path, file_name)
            L1.append([path[6:], imgName])
            # print(file_name)
    data = pd.DataFrame(L1, columns=["Name", "ImagePath"])
    class_num = np.unique(data["Name"])
    # print((class_num))
    data["Label"] = data["Name"].apply(lambda x: class_num.tolist().index(x))
    return data


if __name__ == '__main__':
    datas = main()
    print(datas[(datas["Label"] == 0)])
