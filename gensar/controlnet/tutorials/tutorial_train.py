from share import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


if __name__ == "__main__":
    # Configs
    resume_path = "./models/control_sd15_ini.ckpt"
    batch_size = 3
    logger_freq = 100
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model("./models/cldm_v15.yaml").cpu()
    model.load_state_dict(load_state_dict(resume_path, location="cpu"))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=1,
        dirpath="./checkpoints/",
        filename="sample-mnist-{epoch:02d}",
    )
    # Misc
    dataset = MyDataset()
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    # trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])
    trainer = pl.Trainer(gpus=4, precision=32, callbacks=[logger, checkpoint_callback])

    # Train!
    trainer.fit(model, dataloader)
