from torch.utils.data import DataLoader

import dataloader
import numpy as np
import datasetSplite
import demo.dataloader as Da
if __name__ == '__main__':
    df = datasetSplite.main()
    train_loader = DataLoader(dataloader.DataLoader(df, 114), batch_size=129, shuffle=True)
    # train_loader = DataLoader(Da.FacenetDataset(input_shape=(114,114), lines=["ImagePath"], num_classes = len(np.unique(df["Name"])), random =True))
    for idx, (img, label) in enumerate(train_loader):
        print("Idx:%d\n" % idx)
        print(img.shape)
        print(label.shape)
