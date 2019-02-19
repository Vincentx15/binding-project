# Torch imports
import torch
import torch.optim as optim
from torch.utils.data import Subset, DataLoader

# Outside packages
import os
import numpy as np
import time

# Homemade modules
from models.C3D import C3D
from data.dataset_loader import Conv3DDataset
from src.utils import Tensorboard, mkdirs
import src.learning as learn

'''
Dataloader creation
'''
dataset = Conv3DDataset(pocket_path='data/pockets/sample/', ligand_path='data/ligands/whole_dict_embed_128.p')

n = len(dataset)
indices = list(range(n))
split = 0.8
batch_size = 32

train_indices = indices[:int(split * n)]
valid_indices = indices[int(split * n):]

train_set = Subset(dataset, train_indices)
valid_set = Subset(dataset, valid_indices)

train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(dataset=valid_set, shuffle=True, batch_size=batch_size)

'''
Model loading
'''
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = C3D()
model.to(device)

'''
Optimizer instanciation
'''

criterion = torch.nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

'''
Experiment Setup
'''
name = 'First_test'
log_folder, result_folder = mkdirs(name)
writer = Tensorboard(log_folder)

'''
Run
'''
learn.train_model(model=model,
                  criterion=criterion,
                  optimizer=optimizer,
                  device=device,
                  train_loader=train_loader,
                  validation_loader=valid_loader,
                  save_path=result_folder,
                  writer=writer,
                  num_epochs=5,
                  wall_time=.25)

'''
configuration = {
        'name': name,
        'split': split,
        'size': size,
        'random': random,
        'seed': seed,
        'n_epochs': n_epochs,
        'wall_time': wall_time,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'log_interval': log_interval
    }    

    try:
        os.mkdir(result_folder)
    except FileExistsError:
        pass

    with open(result_folder + 'config.json', 'w') as f:
        json.dump(configuration, f)


    best_model = training.train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        dataloaders=dataloaders,
        writer=writer,
        num_epochs=n_epochs,
        log_interval=log_interval,
        wall_time=wall_time,
        save=save
    )

    torch.save(best_model, result_folder + 'best_model.pth')
'''
