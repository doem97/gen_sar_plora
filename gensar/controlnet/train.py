from share import setup_config

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

# from dataset_fusar_V1 import MyDataset as FUSAR_V1_Dataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import argparse

setup_config()


def main():
    parser = argparse.ArgumentParser(description="Training script")

    # Configs
    parser.add_argument(
        "--resume_path",
        default="./models/control_sd15_ini.ckpt",
        help="Path to the checkpoint to resume training from",
    )
    parser.add_argument(
        "--batch_size", type=int, default=24, help="Batch size for training"
    )
    parser.add_argument(
        "--logger_freq", type=int, default=100, help="Logging frequency"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate"
    )
    parser.add_argument("--sd_locked", action="store_true", help="Lock state_dict")
    parser.add_argument(
        "--only_mid_control", action="store_true", help="Only use mid control"
    )
    parser.add_argument("--gpus", type=int, default=4, help="gpus to use")
    parser.add_argument(
        "--dataset",
        choices=["fusar_v1", "fusrs_cam_v1", "fusrs_cam_v2"],
        default="fusrs_cam_v1",
        help="Choose the dataset to use for training",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./checkpoints/debug/",
        help="Choose the dataset to use for training",
    )

    args = parser.parse_args()

    model = create_model("./models/cldm_v15.yaml").cpu()
    model.load_state_dict(load_state_dict(args.resume_path, location="cpu"))
    model.learning_rate = args.learning_rate
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=1,
        dirpath=args.ckpt_path,
        filename="epoch_{epoch}",
        save_top_k=-1,  # Save all checkpoints
    )

    # Load dataset
    if args.dataset == "fusar_v1":
        from dataset_fusar_V1 import MyDataset
    elif args.dataset == "fusrs_cam_v1":
        from dataset_fusrs_cam import MyDataset
    elif args.dataset == "fusrs_cam_v2":
        from dataset_fusrs_cam_v2 import MyDataset
    dataset = MyDataset()
    dataloader = DataLoader(
        dataset, num_workers=8, batch_size=args.batch_size, shuffle=True
    )
    logger = ImageLogger(batch_frequency=args.logger_freq)
    trainer = pl.Trainer(
        gpus=args.gpus, precision=32, callbacks=[logger, checkpoint_callback]
    )

    # Train!
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
