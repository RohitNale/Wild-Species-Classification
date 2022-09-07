"""
python src/train_efficientnet_b5.py --epochs 1 --batch_size 32 --model_ckpts "../model_ckpts/efficientnet_b5" --dry_run
"""
import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

FILE = Path(__file__).resolve()
for file in list(FILE.parents)[:2]:
    if str(file) not in sys.path:
        sys.path.append(str(file))  # add ROOT to PATH

from src.utils import seed_everything, create_folder, convert_time
from src.utils import save_model, SaveBestModel
from datamodule import data_loader
from src.models import build_efficientnet_b5, load_checkpoint


def sanity_check(model, dataloader, criterion, optimizer, epochs=10):
    print("Sanity Check", "-" * 50)

    batch = next(iter(dataloader))
    xs, ys = batch["image"], batch["label"]
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        outputs = model(xs)
        loss = criterion(outputs, ys)
        loss.backward()
        optimizer.step()
        print(f"{epoch} - loss: {loss:.4f}")


def fit(args, epoch, model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0
    pbar = tqdm(dataloader, total=len(dataloader), leave=False)
    for batch in pbar:
        xs, ys = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad()

        logits = model.forward(xs)
        loss = criterion(logits, ys)

        predicted = torch.nn.functional.softmax(logits, dim=1)
        predicted = torch.argmax(predicted, 1)

        pbar.set_description_str(desc=f"Epoch {epoch + 1}")
        pbar.set_postfix_str(f"loss: {loss:.4f}")

        loss.backward()
        optimizer.step()

        total += ys.size(0)
        running_loss += loss.item()
        running_correct += predicted.eq(torch.argmax(ys, 1)).sum().item()

        # if args.dry_run:
        #     break

    train_loss = running_loss / len(dataloader)
    accu = 100. * running_correct / total
    return train_loss, accu


def validate(args, epoch, model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader), leave=False)
        for batch in pbar:
            xs, ys = batch["image"].to(device), batch["label"].to(device)
            logits = model.forward(xs)
            loss = criterion(logits, ys)
            predicted = torch.nn.functional.softmax(logits, dim=1)
            predicted = torch.argmax(predicted, 1)

            pbar.set_description_str(desc=f"Epoch {epoch + 1}")
            pbar.set_postfix_str(f"loss: {loss:.4f}")

            total += ys.size(0)
            running_loss += loss.item()
            running_correct += predicted.eq(torch.argmax(ys, 1)).sum().item()

            # if args.dry_run:
            #     break

        val_loss = running_loss / len(dataloader)
        val_accuracy = 100.0 * running_correct / total
        return val_loss, val_accuracy


def cf_matrix(model, dataloader, species_labels, device, num_classes=8):
    y_pred = []
    y_true = []

    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader), leave=False, desc=f"Confusion Matrix")
        for batch in pbar:
            xs, ys = batch["image"].to(device), batch["label"].to(device)
            logits = model.forward(xs)
            predicted = torch.nn.functional.softmax(logits, dim=1)
            predicted = torch.argmax(predicted, 1)
            ys = torch.argmax(ys, 1).cpu().numpy()
            y_pred.extend(predicted)  # Save Prediction
            y_true.extend(ys)  # Save Truth

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10,
                         index=[i for i in species_labels],
                         columns=[i for i in species_labels])
    plt.figure(figsize=(12, 7))
    plt.savefig("./model_ckpts/efficientnet_b5/output.png")

    # To get the per-class accuracy:
    # print(confusionMatrix.diag() / confusionMatrix.sum(1))


def main(args):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computation device: {device}\n")

    dataset_kwargs = {'data_dir': args.data_dir,
                      'batch_size': args.batch_size,
                      }

    if device == "cuda":
        cuda_kwargs = {'num_workers': os.cpu_count(),
                       'pin_memory': True,
                       }
        dataset_kwargs.update(cuda_kwargs)

    train_loader, eval_loader, _, species_labels = data_loader(**dataset_kwargs)

    # initialize the model
    model = build_efficientnet_b5(device=device, fine_tune=True, num_classes=8)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.08317637711026708, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=1)

    save_best_model = SaveBestModel()  # initialize SaveBestModel class

    best_valid_loss = float('inf')  # check best valid loss

    # sanity_check(model, train_loader, criterion, optimizer, epochs=10)
    logger = SummaryWriter(log_dir=f'../log_runs/efficientnet_b5')

    print(f"[INFO]: Run {args.epochs} Epochs")
    # create output format
    print("{:^6}{:^11}{:^10}{:^11}{:^10}".format(
        'epoch', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc'))
    print("{:^6}{:^11}{:^10}{:^11}{:^10}".format(
        '-' * len('epoch'), '-' * len('train_loss'), '-' * len('train_acc'), '-' * len('valid_loss'),
        '-' * len('valid_acc')))

    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        train_loss, train_acc = fit(args, epoch, model, train_loader, criterion, optimizer, device)
        eval_loss, eval_acc = validate(args, epoch, model, eval_loader, criterion, device)
        # scheduler.step()

        # if args.dry_run:
        #     break

        logger.add_scalar("Loss/train", train_loss, epoch + 1)
        logger.add_scalar("Accuracy/train", train_acc, epoch + 1)
        logger.add_scalar("Loss/eval", eval_loss, epoch + 1)
        logger.add_scalar("Accuracy/eval", eval_acc, epoch + 1)

        # logging
        if eval_loss < best_valid_loss:
            best_valid_loss = eval_loss
            print("{:^6}{:^11.4f}{:^10.4f}{:^11.4f}{:^10.4f}{}".format(
                epoch + 1, train_loss, train_acc, eval_loss, eval_acc, " Save Best Model"
            ))
            # save the best model till now if we have the least loss in the current epoch
            save_best_model(args, eval_loss, epoch, model, optimizer, criterion)
        else:
            print("{:^6}{:^11.4f}{:^10.4f}{:^11.4f}{:^10.4f}".format(
                epoch + 1, train_loss, train_acc, eval_loss, eval_acc))

    if not args.dry_run:
        # save last model checkpoint
        save_model(args, model, optimizer, criterion)

    print('TRAINING COMPLETE - {}'.format(convert_time(int(time.perf_counter() - t0))))
    logger.flush()
    logger.close()


def resume(args):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computation device: {device}\n")

    dataset_kwargs = {'data_dir': args.data_dir,
                      'batch_size': args.batch_size,
                      }

    if device == "cuda":
        cuda_kwargs = {'num_workers': os.cpu_count(),
                       'pin_memory': True,
                       }
        dataset_kwargs.update(cuda_kwargs)

    train_loader, eval_loader, _, species_labels = data_loader(**dataset_kwargs)
    # initialize the model
    model = build_efficientnet_b5(device=device, fine_tune=True, num_classes=8)
    ckpt_path = f"../model_ckpts/efficientnet_b5/best_model.pth"
    optimizer = torch.optim.SGD(model.parameters(), lr=0.08317637711026708, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=1)
    p_epochs, model_ckpt, criterion, optimizer_ckpt = load_checkpoint(ckpt_path, model, optimizer)

    cf_matrix(model, eval_loader, species_labels, device, num_classes=8)
    save_best_model = SaveBestModel()  # initialize SaveBestModel class
    best_valid_loss = float('inf')  # check best valid loss
    # sanity_check(model, train_loader, criterion, optimizer, epochs=10)
    logger = SummaryWriter(log_dir=f'../log_runs/efficientnet_b5')

    epochs = p_epochs + args.epochs  # train for more epochs
    print(f"[INFO] Train for {args.epochs} more epochs...")

    t0 = time.perf_counter()
    for epoch in range(epochs):
        train_loss, train_acc = fit(args, model_ckpt, model, train_loader, criterion, optimizer_ckpt, device)
        eval_loss, eval_acc = validate(args, model_ckpt, model, eval_loader, criterion, device)
        # scheduler.step()

        # if args.dry_run:
        #     break

        logger.add_scalar("Loss/train", train_loss, epoch + 1)
        logger.add_scalar("Accuracy/train", train_acc, epoch + 1)
        logger.add_scalar("Loss/eval", eval_loss, epoch + 1)
        logger.add_scalar("Accuracy/eval", eval_acc, epoch + 1)

        # logging
        if eval_loss < best_valid_loss:
            best_valid_loss = eval_loss
            print("{:^6}{:^11.4f}{:^10.4f}{:^11.4f}{:^10.4f}{}".format(
                epoch + 1, train_loss, train_acc, eval_loss, eval_acc, " Save Best Model"
            ))
            # save the best model till now if we have the least loss in the current epoch
            save_best_model(args, eval_loss, epoch, model, optimizer, criterion)
        else:
            print("{:^6}{:^11.4f}{:^10.4f}{:^11.4f}{:^10.4f}".format(
                epoch + 1, train_loss, train_acc, eval_loss, eval_acc))

    if not args.dry_run:
        save_model(args, model, optimizer, criterion)  # save last model checkpoint

    print('TRAINING COMPLETE - {}'.format(convert_time(int(time.perf_counter() - t0))))
    logger.flush()
    logger.close()


def test(args):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computation device: {device}\n")

    dataset_kwargs = {'data_dir': args.data_dir,
                      'batch_size': args.batch_size,
                      }

    if device == "cuda":
        cuda_kwargs = {'num_workers': os.cpu_count(),
                       'pin_memory': True,
                       }
        dataset_kwargs.update(cuda_kwargs)

    _, _, test_loader, species_labels = data_loader(**dataset_kwargs)

    # initialize the model
    model = build_efficientnet_b5(fine_tune=True, num_classes=8)
    model = model.to(device)

    # total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.\n")

    model_path = f"./model_ckpts/efficientnet_b5/last_model.pth"
    # load the model checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("[INFO] model loaded...")
    except Exception as e:
        print(f"[Error] Model not found in {model_path}\n{e}")

    preds_collector = []

    # put the model in eval mode, so we don't update any parameters
    model.eval()

    # we aren't updating our weights so no need to calculate gradients
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            xs = batch["image"].to(device)
            # run the forward step
            logits = model.forward(xs)
            # apply softmax so that model outputs are in range [0,1]
            preds = torch.nn.functional.softmax(logits, dim=1)
            # store this batch's predictions in df note that PyTorch Tensors need to first be detached from their
            # computational graph before converting to numpy arrays

            preds_df = pd.DataFrame(
                preds.detach().numpy(),
                index=batch["image_id"],
                columns=species_labels,
            )
            preds_collector.append(preds_df)

    if not args.dry_run:
        submission_df = pd.concat(preds_collector)
        submission_file = os.path.join(args.data_dir, "submission_format.csv")
        submission_format = pd.read_csv(submission_file, index_col="id")
        assert all(submission_df.index == submission_format.index)
        assert all(submission_df.columns == submission_format.columns)
        submission_df.to_csv("submissions/submission_df_efficientnet_b5.csv")


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="Conser-vision Image Classification")
    parser.add_argument('--data_dir', type=str, default='dataset', metavar='P',
                        help='Path for dataset')
    parser.add_argument('--model_ckpts', type=str, default='model_ckpts', metavar='P',
                        help='Path For Saving the current Model')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--test', action='store_true', default=False,
                        help='check model on test data')

    opt = parser.parse_args()

    # for i in vars(opt): print(f"{i:>12}", ':', vars(opt)[i])

    create_folder(opt.model_ckpts)
    create_folder('../submissions')

    if opt.resume:
        resume(opt)
    else:
        main(opt)

    if opt.test:
        test(opt)
