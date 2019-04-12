import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--parallel", help="decide if we run thing on parallel", action='store_true')
parser.add_argument("-d", "--data_loading", default='hard', choices=['fly', 'hard', 'ram', 'hram'],
                    help="choose the way to load data")
parser.add_argument("-bs", "--batch_size", type=int, default=128,
                    help="choose the batch size")
parser.add_argument("-nw", "--workers", type=int, default=24,
                    help="Number of workers to load data")
parser.add_argument("-n", "--name", type=str, default='default_name',
                    help="Name for the logs")
args = parser.parse_args()

# Torch imports
import torch
import torch.optim as optim

import time

# Homemade modules
from src.utils import Tensorboard, mkdirs
import src.learning as learn
from data.loader import Loader

from models.SmallC3D import SmallC3D

# from models.Toy import Toy
# from models.C3D import C3D

'''
Hardware settings
'''

torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# This is to create an appropriate number of workers, but works too with cpu
if args.parallel:
    used_gpus_count = torch.cuda.device_count()
else:
    used_gpus_count = 1

print('Done importing')

'''
Dataloader creation
'''

if args.data_loading == 'fly':
    augment_flips = True
    ram = False
    pocket_path = 'data/pockets/unique_pockets'
elif args.data_loading == 'hard':
    augment_flips = False
    ram = False
    pocket_path = 'data/pockets/unique_pockets_hard'
elif args.data_loading == 'ram':
    augment_flips = True
    ram = True
    pocket_path = 'data/pockets/unique_pockets'
elif args.data_loading == 'hram':
    augment_flips = False
    ram = True
    pocket_path = 'data/pockets/unique_pockets_hard'
else:
    raise ValueError('Not implemented this DataLoader yet')

batch_size = args.batch_size
num_workers = args.workers

loader = Loader(pocket_path=pocket_path, ligand_path='data/ligands/whole_dict_embed_128.p',
                batch_size=batch_size, num_workers=num_workers,
                augment_flips=augment_flips, ram=ram)
train_loader, _, test_loader = loader.get_data()
print('Created data loader')

# a = time.perf_counter()
# for batch_idx, (inputs, labels) in enumerate(train_loader):
#     if not batch_idx % 20:
#         print(batch_idx, time.perf_counter() - a)
#         a = time.perf_counter()
# print('Done in : ', time.perf_counter() - a)

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
name = args.name
log_folder, result_folder = mkdirs(name)
writer = Tensorboard(log_folder)

'''
Get Summary of the model
'''

# from torchsummary import summary
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
                  test_loader=test_loader,
                  save_path=result_folder,
                  writer=writer,
                  num_epochs=400,
                  wall_time=9)

'''
Dataloader creation
'''
# if args.data_loading == 'hard':
#     from data.dataset_loader_hard import get_data
# elif args.data_loading == 'ram':
#     from data.dataset_loader_ram import get_data
# elif args.data_loading == 'hram':
#     from data.dataset_loader_hardram import get_data
# else:
#     from data.dataset_loader import get_data

# batch_size = 8
# # batch_size = args.batch_size
# # num_workers = 6 * used_gpus_count
# num_workers = 20
# train_loader, valid_loader, test_loader = get_data(batch_size=batch_size, num_workers=num_workers)
# print('Created data loader')
