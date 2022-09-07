import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

FILE = Path(__file__).resolve()
for file in list(FILE.parents)[:2]:
    if str(file) not in sys.path:
        sys.path.append(str(file))  # add ROOT to PATH

from src.utils import seed_everything, create_folder, convert_time
from datamodule import data_loader
from src.models import build_resnet50, load_checkpoint


def main(args):
    print("Running Test...")
    seed_everything(42)
    folder = './submissions'
    create_folder(folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computation device: {device}\n")

    dataset_kwargs = {'data_dir': args.data_dir,
                      'batch_size': args.batch_size,
                      'num_workers': 0
                      }

    if device == "cuda":
        cuda_kwargs = {'num_workers': os.cpu_count(),
                       'pin_memory': True,
                       }
        dataset_kwargs.update(cuda_kwargs)

    _, _, test_loader, species_labels = data_loader(**dataset_kwargs)

    # initialize the model
    model = build_resnet50(device=device, fine_tune=True, num_classes=8)
    model = model.to(device)
    model_path = os.path.join(f"ckpts/{args.ckpts}/model","best_model.pth")
    _, model_ckpt, _, _ = load_checkpoint(model_path, model, device)

    preds_collector = []
    model_ckpt.eval()  # put the model in eval mode, so we don't update any parameters

    t0 = time.perf_counter()
    # we aren't updating our weights so no need to calculate gradients
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            xs = batch["image"].to(device)
            logits = model_ckpt.forward(xs)
            preds = torch.nn.functional.softmax(logits, dim=1)

            preds_df = pd.DataFrame(
                preds.detach().numpy(),
                index=batch["image_id"],
                columns=species_labels,
            )
            preds_collector.append(preds_df)


    submission_df = pd.concat(preds_collector)
    submission_file = os.path.join(args.data_dir, "submission_format.csv")
    submission_format = pd.read_csv(submission_file, index_col="id")
    assert all(submission_df.index == submission_format.index)
    assert all(submission_df.columns == submission_format.columns)

    submission_df.to_csv(f"ckpts/{args.ckpts}/submission_df.csv")

    print('Create Submission - {}'.format(convert_time(int(time.perf_counter() - t0))))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conser-vision Image Classification")
    parser.add_argument('--data_dir', type=str, default='dataset', metavar='P',
                        help='Path for dataset')
    parser.add_argument('--ckpts', type=str, default='checkpoint', metavar='P',
                        help='Path For Saving the current Model')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')

    opt = parser.parse_args()
    # for i in vars(opt): print(f"{i:>12}", ':', vars(opt)[i])

    main(opt)