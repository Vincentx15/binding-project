import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--parallel", default=True, help="If we don't want to run thing in parallel",
                    action='store_false')
parser.add_argument("-s", "--siamese", default=True, help="If we don't want to use siamese loading",
                    action='store_false')
parser.add_argument("-d", "--data_loading", default='hard', choices=['fly', 'hard', 'ram', 'hram'],
                    help="choose the way to load data")
parser.add_argument("-po", "--pockets", default='unique_pockets_hard',
                    choices=['unique_pockets', 'unique_pockets_hard', 'unaligned', 'unaligned_hard'],
                    help="choose the data to use for the pocket inputs")
parser.add_argument("-bs", "--batch_size", type=int, default=128, help="choose the batch size")
parser.add_argument("-nw", "--workers", type=int, default=20, help="Number of workers to load data")
parser.add_argument("-wt", "--wall_time", type=int, default=None, help="Max time to run the model")
parser.add_argument("-n", "--name", type=str, default='default_name', help="Name for the logs")
args = parser.parse_args()

# Torch imports
import torch
import torch.optim as optim

import time
import os

# Homemade modules
from src.utils import Tensorboard, mkdirs
import src.learning as learn
from data.loader import Loader

from models.BabyC3D import BabyC3D
from models.SmallC3D import SmallC3D

# from models.Toy import Toy
# from models.C3D import C3D

print('Done importing')

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

print(f'Using {used_gpus_count} GPUs')

'''
Dataloader creation
'''

pocket_file = 'data/pockets/'
pocket_data = args.pockets
pocket_path = os.path.join(pocket_file, pocket_data)
print(f'Using {pocket_path} as the pocket inputs')

if args.data_loading == 'fly':
    augment_flips = True
    ram = False
elif args.data_loading == 'hard':
    augment_flips = False
    ram = False
elif args.data_loading == 'ram':
    augment_flips = True
    ram = True
elif args.data_loading == 'hram':
    augment_flips = False
    ram = True
else:
    raise ValueError('Not implemented this DataLoader yet')

# batch_size = 8
batch_size = args.batch_size
num_workers = args.workers
siamese = args.siamese

print(f'Using batch_size of {batch_size}, {"siamese" if siamese else "serial"} loading')

loader = Loader(pocket_path=pocket_path, ligand_path='data/ligands/whole_dict_embed_128.p',
                batch_size=batch_size, num_workers=num_workers, siamese=siamese,
                augment_flips=augment_flips, ram=ram)
train_loader, _, test_loader = loader.get_data()

print('Created data loader')

if len(train_loader) == 0 & len(test_loader) == 0:
    raise ValueError('there are not enough points compared to the BS')

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
# model = SmallC3D()
model = BabyC3D()
model.to(device)

print(f'Using {model.__class__} as model')

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

print(f'Saving result in {name}')


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

wall_time = args.wall_time

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
                  wall_time=wall_time)

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
