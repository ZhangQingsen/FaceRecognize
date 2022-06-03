from torch.utils.data import DataLoader

import datasetSplite
import run
import dataloader


def main(curr_epoch, epoch_step, batch_size=9):
    df = datasetSplite.main()
    run.run(df, df, curr_epoch, epoch_step, batch_size)
    pass


if __name__ == "__main__":
    main(0, 1, batch_size=129)
