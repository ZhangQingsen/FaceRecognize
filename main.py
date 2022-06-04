from torch.utils.data import DataLoader

import datasetSplite
import run
import dataloader


def main(curr_epoch, epoch_step, batch_size=9, split_rate=0.7, model_load_dir=''):
    # split_rate 建议设置 0.5-0.8

    df = datasetSplite.main()
    df_train = df
    # df_train = df.random_state(frac=split_rate)
    df_test = df_train

    if len(model_load_dir):
        model_load=111
    else:
        model_load=None
    run.run(df, curr_epoch, epoch_step, batch_size,split_rate=split_rate, model_load_dir=model_load_dir)
    pass


if __name__ == "__main__":
    
    # model_load_dir = './model_data/ep001-loss771.730-accu0.059.pth'
    model_load_dir = ''
    main(0, 10, batch_size=45, split_rate=0.7, model_load_dir=model_load_dir)
