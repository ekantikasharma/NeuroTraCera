import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm


# ============================================================
# WINDOWS UTF-8 FIX
# ============================================================

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "Training"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "Testing"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "brain_tumor_model.pth"
)

CHECKPOINT_PATH = os.path.join(
    MODEL_DIR,
    "checkpoint.pth"
)


# ============================================================
# CPU-FRIENDLY SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

# Keep this at 16 first.
BATCH_SIZE = 16

# Number of additional epochs.
EPOCHS = 5

LEARNING_RATE = 0.0001

# Windows + CPU: 0 is safest.
NUM_WORKERS = 0


# ============================================================
# FINAL CLASS ORDER USED BY YOUR BACKEND
# ============================================================

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "Pituitary",
    "No Tumor"
]

NUM_CLASSES = 4


# ============================================================
# CPU THREAD CONTROL
# ============================================================

cpu_count = os.cpu_count() or 4

torch.set_num_threads(
    max(
        1,
        min(cpu_count // 2, 4)
    )
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("XAI BRAIN TUMOR - CPU OPTIMIZED TRAINING")
print("=" * 70)

print("Device:", DEVICE)
print("CPU threads:", torch.get_num_threads())
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Workers:", NUM_WORKERS)


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(
        f"Training folder not found:\n{TRAIN_DIR}"
    )

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(
        f"Testing folder not found:\n{TEST_DIR}"
    )

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


test_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)


# ImageFolder automatically sorts folders alphabetically.
#
# Your folders:
#
# 0 = Glioma
# 1 = Meningioma
# 2 = No Tumor
# 3 = Pituitary
#
# But your backend uses:
#
# 0 = Glioma
# 1 = Meningioma
# 2 = Pituitary
# 3 = No Tumor
#
# Therefore we remap labels 2 and 3.


print("\nOriginal ImageFolder mapping:")

print(
    train_dataset.class_to_idx
)

print(
    "\nTraining images:",
    len(train_dataset)
)

print(
    "Testing images:",
    len(test_dataset)
)


# ============================================================
# LABEL REMAPPING
# ============================================================

def remap_label(label):

    # ImageFolder:
    # 0 Glioma
    # 1 Meningioma
    # 2 No Tumor
    # 3 Pituitary

    # Application:
    # 0 Glioma
    # 1 Meningioma
    # 2 Pituitary
    # 3 No Tumor

    if label == 2:
        return 3

    if label == 3:
        return 2

    return label


# ============================================================
# CUSTOM DATASET WRAPPER
# ============================================================

class RemappedDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset
    ):

        self.dataset = dataset

    def __len__(self):

        return len(
            self.dataset
        )

    def __getitem__(
        self,
        index
    ):

        image, label = self.dataset[index]

        label = remap_label(
            label
        )

        return image, label


train_dataset = RemappedDataset(
    train_dataset
)

test_dataset = RemappedDataset(
    test_dataset
)


print(
    "\nClass mapping used for training:"
)

for i, name in enumerate(
    CLASS_NAMES
):

    print(
        f"{i} = {name}"
    )


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=False
)

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=False
)


print(
    "\nTraining batches:",
    len(train_loader)
)

print(
    "Testing batches:",
    len(test_loader)
)


# ============================================================
# CREATE RESNET50
# ============================================================

print("\n" + "=" * 70)
print("CREATING RESNET50")
print("=" * 70)

model = models.resnet50(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)


# ============================================================
# LOAD EXISTING TRAINED MODEL
# ============================================================

if not os.path.exists(
    MODEL_PATH
):

    raise FileNotFoundError(
        "\nExisting model not found:\n"
        + MODEL_PATH
    )


print(
    "\nExisting model found:"
)

print(
    MODEL_PATH
)


old_checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)


# ============================================================
# EXTRACT STATE DICT
# ============================================================

if isinstance(
    old_checkpoint,
    dict
):

    if "state_dict" in old_checkpoint:

        state_dict = (
            old_checkpoint[
                "state_dict"
            ]
        )

    elif "model_state_dict" in old_checkpoint:

        state_dict = (
            old_checkpoint[
                "model_state_dict"
            ]
        )

    else:

        state_dict = old_checkpoint

else:

    state_dict = old_checkpoint


# ============================================================
# REMOVE MODULE PREFIX
# ============================================================

cleaned_state_dict = {}

for key, value in state_dict.items():

    if key.startswith(
        "module."
    ):

        key = key.replace(
            "module.",
            "",
            1
        )

    cleaned_state_dict[key] = value


# ============================================================
# CHECK FINAL FC LAYER
# ============================================================

fc_weight = cleaned_state_dict.get(
    "fc.weight"
)

fc_bias = cleaned_state_dict.get(
    "fc.bias"
)


if fc_weight is None:

    raise RuntimeError(
        "fc.weight was not found in the model."
    )

if fc_bias is None:

    raise RuntimeError(
        "fc.bias was not found in the model."
    )


# ============================================================
# FIX OLD MODEL CLASS ORDER
# ============================================================

print(
    "\nCorrecting old model class mapping..."
)

print(
    "Old model:"
)

print(
    "0 = Glioma"
)

print(
    "1 = Meningioma"
)

print(
    "2 = No Tumor"
)

print(
    "3 = Pituitary"
)


print(
    "\nNew model:"
)

print(
    "0 = Glioma"
)

print(
    "1 = Meningioma"
)

print(
    "2 = Pituitary"
)

print(
    "3 = No Tumor"
)


# Swap rows 2 and 3.

new_weight = fc_weight.clone()

new_weight[0] = fc_weight[0]
new_weight[1] = fc_weight[1]
new_weight[2] = fc_weight[3]
new_weight[3] = fc_weight[2]


new_bias = fc_bias.clone()

new_bias[0] = fc_bias[0]
new_bias[1] = fc_bias[1]
new_bias[2] = fc_bias[3]
new_bias[3] = fc_bias[2]


cleaned_state_dict[
    "fc.weight"
] = new_weight

cleaned_state_dict[
    "fc.bias"
] = new_bias


# ============================================================
# LOAD MODEL
# ============================================================

model.load_state_dict(
    cleaned_state_dict,
    strict=True
)


print(
    "\nExisting model loaded successfully."
)

print(
    "Class mapping corrected."
)


model = model.to(
    DEVICE
)


# ============================================================
# FREEZE RESNET50 BACKBONE
# ============================================================

print(
    "\nFreezing ResNet50 backbone..."
)

for parameter in model.parameters():

    parameter.requires_grad = False


# Only final classifier is trainable.

for parameter in model.fc.parameters():

    parameter.requires_grad = True


trainable_parameters = sum(

    p.numel()

    for p in model.parameters()

    if p.requires_grad
)

total_parameters = sum(

    p.numel()

    for p in model.parameters()
)


print(
    "Trainable parameters:",
    f"{trainable_parameters:,}"
)

print(
    "Total parameters:",
    f"{total_parameters:,}"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.fc.parameters(),

    lr=LEARNING_RATE,

    weight_decay=1e-4
)


# ============================================================
# RESUME VARIABLES
# ============================================================

start_epoch = 0

best_accuracy = 0.0


# ============================================================
# LOAD CHECKPOINT IF IT EXISTS
# ============================================================

if os.path.exists(
    CHECKPOINT_PATH
):

    print(
        "\nCheckpoint found."
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )


    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "model_state_dict"
        in checkpoint
    ):

        model.load_state_dict(

            checkpoint[
                "model_state_dict"
            ]
        )


        if (
            "optimizer_state_dict"
            in checkpoint
        ):

            try:

                optimizer.load_state_dict(

                    checkpoint[
                        "optimizer_state_dict"
                    ]

                )

            except Exception as e:

                print(
                    "Could not restore optimizer:",
                    e
                )


        start_epoch = int(
            checkpoint.get(
                "epoch",
                0
            )
        )


        best_accuracy = float(
            checkpoint.get(
                "best_accuracy",
                0.0
            )
        )


        print(
            "Checkpoint loaded."
        )

        print(
            "Last completed epoch:",
            start_epoch
        )

        print(
            "Best accuracy:",
            f"{best_accuracy:.2f}%"
        )

    else:

        print(
            "Checkpoint format is invalid."
        )

else:

    print(
        "\nNo checkpoint found."
    )

    print(
        "Starting from existing model."
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    total = 0

    correct = 0

    running_loss = 0.0


    progress = tqdm(
        train_loader,
        desc="Training",
        leave=True
    )


    for images, labels in progress:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item()
            *
            images.size(0)
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)


        correct += (
            predicted == labels
        ).sum().item()


        accuracy = (
            100.0
            *
            correct
            /
            total
        )


        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{accuracy:.2f}%"
        )


    epoch_loss = (
        running_loss
        /
        total
    )


    epoch_accuracy = (
        100.0
        *
        correct
        /
        total
    )


    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate():

    model.eval()

    total = 0

    correct = 0

    running_loss = 0.0


    with torch.no_grad():

        progress = tqdm(
            test_loader,
            desc="Testing",
            leave=True
        )


        for images, labels in progress:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            running_loss += (
                loss.item()
                *
                images.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            total += labels.size(0)


            correct += (
                predicted == labels
            ).sum().item()


            accuracy = (
                100.0
                *
                correct
                /
                total
            )


            progress.set_postfix(
                acc=f"{accuracy:.2f}%"
            )


    loss = (
        running_loss
        /
        total
    )


    accuracy = (
        100.0
        *
        correct
        /
        total
    )


    return (
        loss,
        accuracy
    )


# ============================================================
# TRAINING START
# ============================================================

print("\n" + "=" * 70)
print("STARTING CPU TRAINING")
print("=" * 70)

print(
    "Existing model:",
    MODEL_PATH
)

print(
    "Checkpoint:",
    CHECKPOINT_PATH
)

print(
    "Starting epoch:",
    start_epoch + 1
)

print(
    "Total epochs:",
    EPOCHS
)

print(
    "Only final FC layer will be trained."
)

print("=" * 70)


# ============================================================
# ALREADY COMPLETE?
# ============================================================

if start_epoch >= EPOCHS:

    print(
        "\nRequested epochs already completed."
    )

    print(
        "To train more, increase EPOCHS."
    )


else:

    for epoch in range(
        start_epoch,
        EPOCHS
    ):

        epoch_number = epoch + 1


        print(
            "\n"
            + "=" * 70
        )

        print(
            f"EPOCH {epoch_number}/{EPOCHS}"
        )

        print(
            "=" * 70
        )


        start_time = time.time()


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, train_accuracy = (
            train_one_epoch()
        )


        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

        test_loss, test_accuracy = (
            evaluate()
        )


        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        elapsed = (
            time.time()
            -
            start_time
        )


        print(
            "\nEpoch completed."
        )

        print(
            "Training Loss:",
            f"{train_loss:.4f}"
        )

        print(
            "Training Accuracy:",
            f"{train_accuracy:.2f}%"
        )

        print(
            "Testing Loss:",
            f"{test_loss:.4f}"
        )

        print(
            "Testing Accuracy:",
            f"{test_accuracy:.2f}%"
        )

        print(
            "Epoch Time:",
            f"{elapsed / 60:.2f} minutes"
        )


        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if test_accuracy > best_accuracy:

            best_accuracy = test_accuracy


            torch.save(
                model.state_dict(),
                MODEL_PATH
            )


            print(
                "\nBEST MODEL SAVED"
            )

            print(
                "Best accuracy:",
                f"{best_accuracy:.2f}%"
            )


        # ====================================================
        # SAVE CHECKPOINT AFTER EVERY EPOCH
        # ====================================================

        checkpoint_data = {

            "epoch":
                epoch_number,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "best_accuracy":
                best_accuracy,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_accuracy,

            "test_loss":
                test_loss,

            "test_accuracy":
                test_accuracy,

            "class_names":
                CLASS_NAMES,

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE

        }


        torch.save(
            checkpoint_data,
            CHECKPOINT_PATH
        )


        print(
            "\nCHECKPOINT SAVED"
        )

        print(
            "Epoch:",
            epoch_number
        )

        print(
            "File:",
            CHECKPOINT_PATH
        )


# ============================================================
# FINISHED
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "TRAINING PROCESS FINISHED"
)

print(
    "=" * 70
)

print(
    "Best Testing Accuracy:",
    f"{best_accuracy:.2f}%"
)

print(
    "\nModel:"
)

print(
    MODEL_PATH
)

print(
    "\nCheckpoint:"
)

print(
    CHECKPOINT_PATH
)

print(
    "\nYou can safely stop the program after"
    " a checkpoint has been saved."
)

print("=" * 70)