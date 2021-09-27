import argparse
import os, sys

import math
import time
import tabulate

import torch
import tqdm
import torch.nn.functional as F

import numpy as np

from subspace_inference import data, models, utils, losses

from experiments.MNIST_SDE_Net import data_loader
from experiments.MNIST_SDE_Net import models
from subspace_inference.posteriors import SWAG_sdenet

from subspace_inference.posteriors.vi_model import VIModel, ELBO
from subspace_inference.posteriors.proj_model import SubspaceModel

import sklearn.decomposition

def nll(outputs, labels):
    labels = labels.astype(int)
    idx = (np.arange(labels.size), labels)
    ps = outputs[idx]
    nll = -np.mean(np.log(ps + 1e-12))
    return nll


parser = argparse.ArgumentParser(description='Subspace VI')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')

parser.add_argument('--dataset_inDomain', default='mnist', help='training dataset')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--test_batch_size', type=int, default=1000)

parser.add_argument('--split_classes', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=0, metavar='N', help='number of workers (default: 4)')
# parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
#                     help='model name (default: None)')

parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs (default: 50')
parser.add_argument('--num_samples', type=int, default=30, metavar='N', help='number of epochs (default: 30')

parser.add_argument('--temperature', type=float, default=1., required=True, 
                    metavar='N', help='Temperature (default: 1.')
parser.add_argument('--no_mu', action='store_true', help='Do not learn the mean of posterior')

parser.add_argument('--rank', type=int, default=5, metavar='N', help='approximation rank (default: 2')
parser.add_argument('--random', action='store_true')

parser.add_argument('--prior_std', type=float, default=1.0, help='std of the prior distribution')
parser.add_argument('--init_std', type=float, default=1.0, help='initial std of the variational distribution')

parser.add_argument('--checkpoint', type=str, default=None, required=True, help='path to SWAG checkpoint')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--bn_subset', type=float, default=1.0, help='BN subset for evaluation (default 1.0)')
parser.add_argument('--max_rank', type=int, default=20, help='maximum rank')

args = parser.parse_args()

class Logger(object):
    def __init__(self,logFile="Default.log"):
        self.terminal = sys.stdout
        self.log = open(logFile,'a')

    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass
sys.stdout = Logger("log_subspace_vi_SDENet2.log")

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('load in-domain data: ',args.dataset_inDomain)
train_loader_inDomain, test_loader_inDomain = data_loader.getDataSet(args.dataset_inDomain,
                                                                     args.batch_size,
                                                                     args.test_batch_size,
                                                                     args.imageSize)
# print("len(test_loader_inDomain).shape={}".format(len(test_loader_inDomain)))
print('==> Building model..')
model = models.SDENet_MNIST(layer_depth=6, num_classes=10, dim=64)
model = model.to(device)

swag_model = SWAG_sdenet(
    model,
    subspace_type='pca',
    subspace_kwargs={
        'max_rank': args.max_rank,
        'pca_rank': args.rank,
    }

)
swag_model.to(device)

print('Loading: %s' % args.checkpoint)
ckpt = torch.load(args.checkpoint)
swag_model.load_state_dict(ckpt, strict=False)

swag_model.set_swa()
swag_model.eval()

mean, var, cov_factor = swag_model.get_space()

print(torch.norm(cov_factor, dim=1))
#print(var)

if args.random:
    #np.linalg.norm表示求范数，默认是二范数L2=sqrt(x1^2+...xn^2); 下面scale为cov_factor的（第一行+第二行）/2
    scale = 0.5 * (np.linalg.norm(cov_factor[1, :]) + np.linalg.norm(cov_factor[0, :]))
    print(scale)
    np.random.seed(args.seed)
    cov_factor = np.random.randn(*cov_factor.shape)

    tsvd = sklearn.decomposition.TruncatedSVD(n_components=args.rank, n_iter=7, random_state=args.seed)
    tsvd.fit(cov_factor)

    cov_factor = tsvd.components_ # self.components_ = VT
    cov_factor /= np.linalg.norm(cov_factor, axis=1, keepdims=True)
    cov_factor *= scale

    print(cov_factor[:, 0])

    cov_factor = torch.FloatTensor(cov_factor, device=mean.device)

vi_model = VIModel(
    subspace=SubspaceModel(mean.cuda(), cov_factor.cuda()),
    init_inv_softplus_sigma=math.log(math.exp(args.init_std) - 1.0),
    prior_log_sigma=math.log(args.prior_std),
    base=model,
    with_mu=not args.no_mu,
)

vi_model = vi_model.cuda()

elbo = ELBO(losses.cross_entropy,
            len(train_loader_inDomain)*args.batch_size,
            args.temperature)

#optimizer = torch.optim.Adam([param for param in vi_model.parameters()], lr=0.01)
optimizer = torch.optim.SGD([param for param in vi_model.parameters()], lr=args.lr, momentum=0.9)

#printf, logfile = utils.get_logging_print(os.path.join(args.dir, args.log_fname + '-%s.txt'))
#print('Saving logs to: %s' % logfile)
columns = ['ep', 'acc', 'loss', 'kl', 'nll', 'sigma_1', 'time']

epoch = 0
for epoch in range(args.epochs):
    time_ep = time.time()
    train_res = utils.train_epoch(train_loader_inDomain, vi_model, elbo, optimizer) #loader中的数据会给到vi_model中

    time_ep = time.time() - time_ep
    #softplus(x) = log(1+exp(x)); vi_model.inv_softplus_sigma=-3
    sigma_1 = torch.nn.functional.softplus(vi_model.inv_softplus_sigma.detach().cpu())[0].item()
    values = ['%d/%d' % (epoch + 1, args.epochs), train_res['accuracy'], train_res['loss'],
              train_res['stats']['kl'], train_res['stats']['nll'], sigma_1, time_ep]
    if epoch == 0:
        print(tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f'))
    else:
        print(tabulate.tabulate([values], columns, tablefmt='plain', floatfmt='8.4f').split('\n')[1])

print("sigma:", torch.nn.functional.softplus(vi_model.inv_softplus_sigma.detach().cpu()))
if not args.no_mu:
    print("mu:", vi_model.mu.detach().cpu().data)

utils.save_checkpoint(
    args.dir,
    epoch,
    name='vi',
    state_dict=vi_model.state_dict()
)

num_samples = args.num_samples

ens_predictions = np.zeros((len(test_loader_inDomain.dataset), 10))
targets = np.zeros(len(test_loader_inDomain.dataset))


columns = ['iter ens', 'acc', 'nll']

for i in range(num_samples): #number of epochs
    with torch.no_grad():
        w = vi_model.sample()
        offset = 0
        for param in model.parameters():
            param.data.copy_(w[offset:offset+param.numel()].view(param.size()).to(args.device))
            offset += param.numel()

    # utils.bn_update(loaders['train'], eval_model, subset=args.bn_subset)

    pred_res = utils.predict(test_loader_inDomain, model)
    ens_predictions += pred_res['predictions']
    targets = pred_res['targets']

    values = ['%3d/%3d' % (i + 1, num_samples),
              np.mean(np.argmax(ens_predictions, axis=1) == targets),
              nll(ens_predictions / (i + 1), targets)]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if i == 0:
        print(table)
    else:
        print(table.split('\n')[2])

ens_predictions /= num_samples
ens_acc = np.mean(np.argmax(ens_predictions, axis=1) == targets)
ens_nll = nll(ens_predictions, targets)

np.savez(
    os.path.join(args.dir, 'ens.npz'),
    seed=args.seed,
    ens_predictions=ens_predictions,
    targets=targets,
    ens_acc=ens_acc,
    ens_nll=ens_nll
)
