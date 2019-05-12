import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--parallel", default=True, help="If we don't want to run thing in parallel",
                    action='store_false')
parser.add_argument("-s", "--siamese", default=False, help="If we want to use siamese loading",
                    action='store_true')
parser.add_argument("-fs", "--full_siamese", default=False, help="If we want to use the full_siamese loading",
                    action='store_true')
parser.add_argument("--shuffled", default=False, help="If we want to use shuffled labels",
                    action='store_true')
parser.add_argument("-d", "--data_loading", default='hard', choices=['fly', 'hard', 'ram', 'hram'],
                    help="choose the way to load data")
parser.add_argument("-po", "--pockets", default='unique_pockets_hard',
                    choices=['unique_pockets', 'unique_pockets_hard', 'unaligned', 'unaligned_hard'],
                    help="choose the data to use for the pocket inputs")
parser.add_argument("-m", "--model", default='baby',
                    choices=['baby', 'small', 'se3cnn', 'c3d', 'small_siamese', 'babyse3cnn'],
                    help="choose the model")
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
from src.utils import Tensorboard, mk_log_trained_dirs
import src.learning as learn
from data.loader import Loader

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
shuffled = args.shuffled
siamese = args.siamese
full_siamese = args.full_siamese
if full_siamese:
    print(f'Using the full siamese pipeline, with batch size of {batch_size}')
    siamese = False
else:
    print(f'Using batch_size of {batch_size}, {"siamese" if siamese else "serial"} loading')

loader = Loader(pocket_path=pocket_path, ligand_path='data/ligands/whole_dict_embed_128.p',
                batch_size=batch_size, num_workers=num_workers, siamese=siamese, full_siamese=full_siamese,
                augment_flips=augment_flips, ram=ram, shuffled=shuffled)
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

model_choice = args.model
model = 0

if model_choice == 'baby':
    from models.BabyC3D import BabyC3D

    # from models.BabyC3D import Crazy

    model = BabyC3D()
    # model = Crazy()
elif model_choice == 'small':
    from models.SmallC3D import SmallC3D

    model = SmallC3D()
elif model_choice == 'se3cnn':
    from models.Se3cnn import Se3cnn

    model = Se3cnn()
elif model_choice == 'c3d':
    from models.C3D import C3D

    model = C3D()
elif model_choice == 'small_siamese':
    from models.Siamese import BabySiamese, SmallSiamese

    # model = BabySiamese()

    model = SmallSiamese()
elif model_choice == 'babyse3cnn':
    from models.BabySe3cnn import BabySe3cnn

    model = BabySe3cnn()
else:
    # Not possible because of argparse
    raise ValueError('Not a possible model')

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
log_folder, result_folder = mk_log_trained_dirs(name)
writer = Tensorboard(log_folder)

print(f'Saving result in {name}')

'''
Get Summary of the model
'''

# train_loader = iter(train_loader)
# batch_x, batch_y = next(train_loader)
# print(batch_x.size())
# batch_x = batch_x.cuda()
# y = model(batch_x)
# print(y, y.size())
# print('batch_y', batch_y.size())
# raise ValueError
# from torchsummary import summary

# summary(model, (4, 42, 32, 32))

# raise ValueError
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
