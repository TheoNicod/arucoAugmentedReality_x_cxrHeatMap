from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from data import RALO_Dataset
from net import Model

def custom_save_function(trainer, pl_module):
    torch.save(pl_module.state_dict(), "./myModel.pt")

# Step 1: Création de l'instance de RALO_Dataset
dataset = RALO_Dataset(imgpath="./RALO-Dataset/CXR_images_scored/", csvpath="./RALO-Dataset/ralo-dataset-metadata.csv", subset="train", transform=transforms.ToTensor())

# Step 2: Création de l'instance de Model
model = Model(architecture='inception_v3', loss_func=torch.nn.L1Loss())

# Step 3: Entraînement du modèle
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset, batch_size=32)

checkpoint_callback = ModelCheckpoint(monitor='Loss/Val')

checkpoint_callback.save_function = custom_save_function


trainer = Trainer(
    max_epochs=10,
    callbacks=[checkpoint_callback]
)
trainer.fit(model, train_loader, val_loader)



