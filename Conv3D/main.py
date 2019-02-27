# Torch imports
import torch
import torch.optim as optim
from torch.utils.data import Subset, DataLoader

# Outside packages
import os
import numpy as np
import time
import argparse
from torchsummary import summary

# Homemade modules
from data.dataset_loader import Conv3DDataset, get_data
from src.utils import Tensorboard, mkdirs
import src.timed_learning as learn
from models.SmallC3D import SmallC3D
from models.Toy import Toy
from models.C3D import C3D


'''
How many GPUs are we using and should we use them in parallel ?

Experiments on very large networks seem to show that it is not faster to use several ones
'''
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--parallel", help="decide if we run thing on parallel", action='store_true')
args = parser.parse_args()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# This is to create an appropriate number of workers, but works too with cpu
if args.parallel:
    used_gpus_count = torch.cuda.device_count()
else:
    used_gpus_count = 1

'''
Dataloader creation
'''

batch_size = 4
train_loader, valid_loader, test_loader = get_data(batch_size=batch_size, num_gpu=used_gpus_count)

'''
Model loading
'''
# model = Toy()
# model = C3D()
model = SmallC3D()
model.to(device)

if used_gpus_count > 1:
    model = torch.nn.DataParallel(model)

'''
Optimizer instanciation
'''

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters())
# optimizer = optim.SGD(model.parameters(), lr=1)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

'''
Experiment Setup
'''
name = 'Smaller'
log_folder, result_folder = mkdirs(name)
writer = Tensorboard(log_folder)

# train_loader = iter(train_loader)
# print(next(train_loader)[0].shape)
# summary(model, (4, 42, 32, 32))
# for p in model.parameters():
#     print(p.__name__)
#     print(p.numel())
# print(sum(p.numel() for p in model.parameters()))


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
                  num_epochs=400)

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
