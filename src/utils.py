import os
import random
from datetime import datetime
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytz import timezone

plt.style.use("ggplot")


def get_ctime(name):
    ind_time = datetime.now(timezone("Asia/Kolkata")).strftime('%Y%m%d_%H%M')
    return f"{name}_{ind_time}"


def convert_time(seconds):
    """
    Convert seconds to minutes and hours.
    """
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)


def create_folder(filename: str):
    """ Create folder if not exists """
    if not os.path.exists(filename):
        os.makedirs(filename)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, best_valid_loss=float("inf")):
        self.best_valid_loss = best_valid_loss

    def __call__(
            self, args, current_valid_loss, epoch, model, optimizer, criterion,
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            # print(f"\nBest validation loss: {self.best_valid_loss}")
            # print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                },
                f"{args.model_ckpts}/best_model.pth",
            )


def save_model(args, model, optimizer, criterion, to_onnx=None):
    """Function to save the trained model to disk."""
    if to_onnx:
        dummy_input = (1, 1, 224, 224)
        torch.onnx.export(model, dummy_input, f"./{args.model_ckpts}/model.onnx")
    else:
        torch.save(
            {
                "epoch": args.epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": criterion,
            },
            f"./{args.model_ckpts}/last_model.pth",
        )


def save_plots(train_accuracy, train_loss, val_accuracy, val_loss, out_path):
    # accuracy plots
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
    ax1.plot(train_accuracy, color="green", label="train accuracy")
    ax1.plot(val_accuracy, color="blue", label="val accuracy")
    ax1.legend()
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")

    ax2.plot(train_loss, color="orange", label="train loss")
    ax2.plot(val_loss, color="red", label="val loss")
    ax2.legend()
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")

    fig.tight_layout()
    fig.savefig(f"./{out_path}/plot.png")
    # plt.show()


def create_metrices(args, y_true, y_pred, species_labels, title=''):
    cf_matrix(args, y_true, y_pred, species_labels, title)
    classification_report(args, y_true, y_pred, species_labels, title)
    roc_auc(args, y_true, y_pred, species_labels, title)
    precision_recall(args, y_true, y_pred, species_labels, title)


def precision_recall(args, y_true, y_pred, species_labels, title):
    classes = [i for i in range(len(species_labels))]
    n_classes = len(classes)
    y_true = label_binarize(y_true, classes=classes)
    y_pred = label_binarize(y_pred, classes=classes)

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = metrics.precision_recall_curve(y_pred[:, i], y_true[:, i])
        average_precision[i] = metrics.average_precision_score(y_pred[:, i], y_true[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(y_pred.ravel(), y_true.ravel())
    average_precision["micro"] = metrics.average_precision_score(y_true, y_pred, average="micro")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    lw = 2
    display = metrics.PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax=ax, lw=lw, name="Micro-average precision-recall", color="gold")

    for i in range(n_classes):
        display = metrics.PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax, lw=lw, name=f"{species_labels[i]}")

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()

    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title(f"Precision Recall - {title}")
    fig.tight_layout()
    fig.savefig(f"{args.model_ckpts}/PRC_{title}.png")
    plt.close()


def roc_auc(args, y_true, y_pred, species_labels, title):
    classes = [i for i in range(len(species_labels))]
    n_classes = len(classes)
    y_true = label_binarize(y_true, classes=classes)
    y_pred = label_binarize(y_pred, classes=classes)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_pred[:, i], y_true[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_pred.ravel(), y_true.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(8, 6), dpi=300)
    lw = 2
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=lw,
                 label=f"{species_labels[i]} (area = {roc_auc[i]:.2f})",
                 )
    plt.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--")
    plt.plot(fpr["micro"], tpr["micro"], color="navy", lw=lw, linestyle="--",
             label=f"{'micro'} (area = {roc_auc['micro']:.2f})",
             )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic - {title}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{args.model_ckpts}/ROC_{title}.png")
    plt.close()


def classification_report(args, y_true, y_pred, species_labels, title=''):
    report = metrics.classification_report(y_true, y_pred, target_names=species_labels, output_dict=True)
    df = pd.DataFrame(report, dtype='float32').transpose()
    df.to_csv(f'{args.model_ckpts}/CR_{title}.csv', index=True)


def cf_matrix(args, y_true, y_pred, species_labels, title=''):
    cm = metrics.confusion_matrix(y_true, y_pred)
    cmp = metrics.ConfusionMatrixDisplay(cm, display_labels=species_labels)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    cmp.plot(ax=ax, values_format='d')
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(f"Confusion Matrix - {title}")
    fig.tight_layout()
    fig.savefig(f"{args.model_ckpts}/CM_{title}.png")
    plt.close()
