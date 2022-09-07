import torch
import torch.nn as nn
import torchvision.models as models


def print_model_params(model):
    # total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.\n")


def load_checkpoint(ckpt_path, model, device, optimizer=None):
    checkpoint = None
    try:  # load the model checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("[INFO] model loaded...")
    except Exception as e:
        print(f"[Error] Model not found in {ckpt_path}\n{e}")

    # initialize optimizer before loading optimizer state_dict
    try:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except Exception as e:
        print(f"[Error] {e}")

    epochs = checkpoint["epoch"]
    criterion = checkpoint["loss"]

    print("[INFO] optimizer loaded...")
    print("[INFO] loss function loaded...")
    print(f"[INFO] Previously trained for {epochs} number of epochs...")
    return epochs, model, criterion, optimizer


def load_model(model, model_type, model_path):
    # load the last model checkpoint
    if model_type == 'last':
        last_model_cp = torch.load(f"{model_path}/last_model.pth")
        model.load_state_dict(last_model_cp['model_state_dict'])

    # load the best model checkpoint
    elif model_type == 'best':
        best_model_cp = torch.load(f"{model_path}/best_model.pth")
        model.load_state_dict(best_model_cp['model_state_dict'])
    return model


def build_resnet50(device, fine_tune=False, num_classes=8):
    print('[INFO]: Loading pre-trained weights')
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # model = load_model(model, model_type=model_type, model_path)

    print('[INFO]: Fine-tuning all layers =', fine_tune)
    for params in model.parameters():
        params.requires_grad = fine_tune
    in_features = model.fc.in_features
    # change the final classification head, it is trainable
    model.fc = nn.Sequential(  # dense layer takes a 2048-dim input and outputs 100-dim
        nn.Linear(in_features, 100), nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
        nn.Dropout(0.1),  # common technique to mitigate overfitting
        nn.Linear(100, num_classes),  # final dense layer outputs 8-dim corresponding to our target classes
    )
    print_model_params(model)
    return model.to(device)


def build_resnet101(device, fine_tune=False, num_classes=8):
    print('[INFO]: Loading pre-trained weights')
    model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    # model = load_model(model, model_type=model_type, model_path)

    print('[INFO]: Fine-tuning all layers =', fine_tune)
    for params in model.parameters():
        params.requires_grad = fine_tune

    # change the final classification head, it is trainable
    model.fc = nn.Sequential(  # dense layer takes a 2048-dim input and outputs 100-dim
        nn.Linear(2048, 100), nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
        nn.Dropout(0.1),  # common technique to mitigate overfitting
        nn.Linear(100, num_classes),  # final dense layer outputs 8-dim corresponding to our target classes
    )
    print_model_params(model)
    return model.to(device)


def build_efficientnet_b5(device, fine_tune=False, num_classes=8):
    print('[INFO]: Loading pre-trained weights')
    model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT)
    # model = load_model(model, model_type=model_type)

    print('[INFO]: Fine-tuning all layers =', fine_tune)
    for params in model.parameters():
        params.requires_grad = fine_tune
    # change the final classification head, it is trainable
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(  # dense layer takes a 2048-dim input and outputs 100-dim
        nn.Dropout(0.4), nn.Linear(in_features, 100), nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
        nn.Dropout(0.1),  # common technique to mitigate overfitting
        nn.Linear(100, num_classes),  # final dense layer outputs 8-dim corresponding to our target classes
    )
    print_model_params(model)
    return model.to(device)


def build_resnext50_32x4d(device, fine_tune=False, num_classes=8):
    print('[INFO]: Loading pre-trained weights')
    model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
    # model = load_model(model, model_type=model_type)

    print('[INFO]: Fine-tuning all layers =', fine_tune)
    for params in model.parameters():
        params.requires_grad = fine_tune

    # change the final classification head, it is trainable
    model.classifier = nn.Sequential(nn.Dropout(0.3),  # dense layer takes a 2048-dim input and outputs 100-dim
                                     nn.Linear(1280, 100), nn.ReLU(inplace=True),
                                     # ReLU activation introduces non-linearity
                                     nn.Dropout(0.1),  # common technique to mitigate overfitting
                                     nn.Linear(100, num_classes))
    print_model_params(model)
    return model.to(device)


def build_inception_v3(device, fine_tune=False, num_classes=8):
    print('[INFO]: Loading pre-trained weights')
    model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    # model = load_model(model, model_type=model_type)

    print('[INFO]: Fine-tuning all layers =', fine_tune)
    for params in model.parameters():
        params.requires_grad = fine_tune

    # change the final classification head, it is trainable
    model.fc = nn.Sequential(  # dense layer takes a 2048-dim input and outputs 100-dim
        nn.Linear(2048, 100), nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
        nn.Dropout(0.1),  # common technique to mitigate overfitting
        nn.Linear(100, num_classes),  # final dense layer outputs 8-dim corresponding to our target classes
    )
    print_model_params(model)
    return model.to(device)
