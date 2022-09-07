import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

FILE = Path(__file__).resolve()
for file in list(FILE.parents)[:2]:
    if str(file) not in sys.path:
        sys.path.append(str(file))  # add ROOT to PATH

from src.utils import seed_everything, create_folder, convert_time
from src.utils import save_model, SaveBestModel, create_metrices
from datamodule import data_loader
from src.models import build_resnet50, load_checkpoint


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

        if args.dry_run:
            break

    train_loss = running_loss / len(dataloader)
    accu = 100. * running_correct / total
    return train_loss, accu


def validate(args, epoch, model, dataloader, species_labels, criterion, device):
    model.eval()
    running_loss = 0.0

    y_pred, y_true = [], []
    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader), leave=False)
        for batch in pbar:
            xs, ys = batch["image"].to(device), batch["label"].to(device)
            logits = model.forward(xs)
            loss = criterion(logits, ys)
            running_loss += loss.item()
            pred = torch.nn.functional.softmax(logits, dim=1)
            pbar.set_description_str(desc=f"Epoch {epoch + 1}")
            pbar.set_postfix_str(f"loss: {loss:.4f}")
            y_pred.extend(torch.argmax(pred, 1))
            y_true.extend(torch.argmax(ys, 1))

            if args.dry_run:
                break

        create_metrices(args, y_true, y_pred, species_labels, title='Resnet50 V')

        val_loss = running_loss / len(dataloader)
        correct = sum(x == y for x, y in zip(y_true, y_pred))
        val_accuracy = 100.0 * correct / len(dataloader.dataset)
        return val_loss, val_accuracy


def train(args):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computation device: {device}\n")

    dataset_kwargs = {'data_dir': args.data_dir,
                      'batch_size': args.batch_size,
                      'num_workers': 0,
                      }

    if device == "cuda":
        cuda_kwargs = {'num_workers': os.cpu_count(),
                       'pin_memory': True,
                       }
        dataset_kwargs.update(cuda_kwargs)

    train_loader, eval_loader, _, species_labels = data_loader(**dataset_kwargs)

    # initialize the model
    model = build_resnet50(device=device, fine_tune=True, num_classes=8)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.08317637711026708, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=1)

    save_best_model = SaveBestModel()  # initialize SaveBestModel class

    best_valid_loss = float('inf')  # check best valid loss

    # sanity_check(model, train_loader, criterion, optimizer, epochs=10)
    log_dir = f'./log_runs/resnet50'
    logger = SummaryWriter(log_dir=log_dir)

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
        eval_loss, eval_acc = validate(args, epoch, model, eval_loader, species_labels, criterion, device)
        # scheduler.step()

        if not args.dry_run:
            logger.add_scalar("Loss/train", train_loss, epoch + 1)
            logger.add_scalar("Accuracy/train", train_acc, epoch + 1)
            logger.add_scalar("Loss/eval", eval_loss, epoch + 1)
            logger.add_scalar("Accuracy/eval", eval_acc, epoch + 1)

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
        logger.flush()
        logger.close()

    print('TRAINING COMPLETE - {}'.format(convert_time(int(time.perf_counter() - t0))))


def resume(args):
    print("Resume Training...")
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computation device: {device}\n")

    dataset_kwargs = {'data_dir': args.data_dir,
                      'batch_size': args.batch_size,
                      'num_workers': os.cpu_count(),
                      }

    if device == "cuda":
        cuda_kwargs = {'num_workers': os.cpu_count(),
                       'pin_memory': True,
                       }
        dataset_kwargs.update(cuda_kwargs)

    train_loader, eval_loader, _, species_labels = data_loader(**dataset_kwargs)
    # initialize the model
    model = build_resnet50(device=device, fine_tune=True, num_classes=8)
    ckpt_path = os.path.join(args.model_ckpts, "last_model.pth")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.08317637711026708, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=1)
    p_epochs, model_ckpt, criterion, optimizer_ckpt = load_checkpoint(ckpt_path, model, device, optimizer)

    save_best_model = SaveBestModel()  # initialize SaveBestModel class
    best_valid_loss = float('inf')  # check best valid loss
    # sanity_check(model, train_loader, criterion, optimizer, epochs=10)

    if not args.dry_run:
        logger = SummaryWriter(log_dir=f'./log_runs/resnet50')

    epochs = p_epochs + args.epochs  # train for more epochs
    print(f"[INFO] Train for {args.epochs} more epochs...")

    if not args.dry_run:
        # create output format
        print("{:^6}{:^11}{:^10}{:^11}{:^10}".format(
            'epoch', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc'))
        print("{:^6}{:^11}{:^10}{:^11}{:^10}".format(
            '-' * len('epoch'), '-' * len('train_loss'), '-' * len('train_acc'), '-' * len('valid_loss'),
            '-' * len('valid_acc')))

    t0 = time.perf_counter()
    for epoch in range(p_epochs, epochs + 1):
        train_loss, train_acc = fit(args, epoch, model, train_loader, criterion, optimizer_ckpt, device)
        eval_loss, eval_acc = validate(args, epoch, model_ckpt, eval_loader, species_labels, criterion, device)
        # scheduler.step()

        if not args.dry_run:
            logger.add_scalar("Loss/train", train_loss, epoch + 1)
            logger.add_scalar("Accuracy/train", train_acc, epoch + 1)
            logger.add_scalar("Loss/eval", eval_loss, epoch + 1)
            logger.add_scalar("Accuracy/eval", eval_acc, epoch + 1)

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
        logger.flush()
        logger.close()

    print('TRAINING COMPLETE - {}'.format(convert_time(int(time.perf_counter() - t0))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conser-vision Image Classification")
    parser.add_argument('--data_dir', type=str, default='dataset', metavar='P',
                        help='Path for dataset')
    parser.add_argument('--model_ckpts', type=str, default='model_ckpts', metavar='P',
                        help='Path For Saving the current Model')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='quickly check a single pass')

    opt = parser.parse_args()
    # for i in vars(opt): print(f"{i:>12}", ':', vars(opt)[i])

    create_folder(opt.model_ckpts)

    if opt.resume:
        resume(opt)
    else:
        train(opt)
