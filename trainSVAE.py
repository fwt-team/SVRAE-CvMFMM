# encoding: utf-8
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: cluster_process.py
@Time: 2020-03-14 15:45
@Desc: cluster_process.py
"""
try:
    import os
    import argparse
    from itertools import chain

    import torch
    import torch.nn as nn
    import numpy as np
    import torch.nn.functional as F

    from sklearn.metrics import silhouette_score as SI, calinski_harabasz_score as CH
    from sklearn.mixture import GaussianMixture
    from vmfmm.vmf import VMFMixture, CVMFMixture
    from sklearn.cluster import KMeans
    from torch.optim.lr_scheduler import StepLR
    from tqdm import tqdm
    from vae.datasets import dataset_list, get_dataloader, get_adhd_data
    from vae.config import RUNS_DIR, Brain_DIR, DEVICE, DATA_PARAMS
    from vae.SVAEmodel import Generator, Encoder
    from vae.utils import init_weights, str2bool
    from cluster_process import ClusterProcess
except ImportError as e:
    print(e)
    raise ImportError


class Trainer:

    def __init__(self, args):

        run_name = args.run_name
        self.dataset_name = args.dataset_name
        self.n_cluster = args.n_cluster

        # make directory
        run_dir = os.path.join(RUNS_DIR, self.dataset_name, run_name, args.version_name)
        self.data_dir = os.path.join(Brain_DIR,  self.dataset_name)
        self.models_dir = os.path.join(run_dir, 'models')
        self.log_path = os.path.join(run_dir, 'logs')

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

        # -----train-----
        # train detail var
        b1 = 0.5
        b2 = 0.99
        decay = 2.5 * 1e-5

        data_params = DATA_PARAMS[self.dataset_name]
        train_batch_size, latent_dim, data_size, train_lr = data_params

        self.n_epochs = args.n_epochs
        self.test_batch_size = 3000
        self.train_batch_size = train_batch_size

        # net
        self.gen = Generator(latent_dim=latent_dim, output_channels=args.input_dim)
        self.encoder = Encoder(input_channels=args.input_dim, output_channels=latent_dim)

        # parallel
        if torch.cuda.device_count() > 1:
            print("this GPU have {} core".format(torch.cuda.device_count()))

        # set device: cuda or cpu
        self.gen.to(DEVICE)
        self.encoder.to(DEVICE)

        # optimization
        self.gen_enc_ops = torch.optim.Adam(chain(
            self.gen.parameters(),
            self.encoder.parameters(),
        ), lr=train_lr, betas=(b1, b2), weight_decay=decay)

        self.lr_s = StepLR(self.gen_enc_ops, step_size=10, gamma=0.95)

    def train(self, data, cp):

        models_dir = self.models_dir
        dataloader = get_dataloader(dataset_path=self.data_dir, dataset_name=self.dataset_name,
                                    batch_size=self.train_batch_size, train=True, data=data)
        test_dataloader = get_dataloader(dataset_path=self.data_dir, dataset_name=self.dataset_name,
                                         batch_size=self.test_batch_size, train=False, data=data)

        # =============================================================== #
        # ============================training=========================== #
        # =============================================================== #
        model_path = os.path.join(models_dir, 'enc.pkl')
        if os.path.exists(model_path):
            self.encoder.load_state_dict(torch.load(model_path, map_location=DEVICE))
            self.test_step(test_dataloader, 0, 0, cp, False)
        else:
            epoch_bar = tqdm(range(0, self.n_epochs))
            for epoch in epoch_bar:
                t_loss = self.train_step(dataloader)
                self.test_step(test_dataloader, epoch, t_loss, cp)

            torch.save(self.gen.state_dict(), os.path.join(models_dir, 'gen.pkl'))
            torch.save(self.encoder.state_dict(), os.path.join(models_dir, 'enc.pkl'))

    def train_step(self, dataloader):

        encoder, gen = self.encoder.train(), self.gen.train()
        gen_enc_ops = self.gen_enc_ops
        L = 0
        for index, x in enumerate(dataloader):
            x = x.to(DEVICE)

            z, mu, kappa, q_z, p_z = encoder(x)
            x_ = gen(z)
            recon_loss = F.mse_loss(x_, x, reduction='sum')

            loss = 15 * recon_loss + torch.distributions.kl.kl_divergence(q_z, p_z).mean()
            L += loss.detach().cpu().numpy()

            gen_enc_ops.zero_grad()
            loss.backward()
            gen_enc_ops.step()

        self.lr_s.step()
        return L / len(dataloader)

    def test_step(self, test_dataloader, epoch, t_loss, cp, save=True):

        gen, encoder = self.gen, self.encoder
        gen.eval()
        encoder.eval()

        with torch.no_grad():
            Z = []
            for index, x in enumerate(test_dataloader):
                x = x.to(DEVICE)

                _z = encoder(x)
                Z.append(_z[0])
            Z = torch.cat(Z, 0)

            _pred, rho = CVMFMixture(n_cluster=self.n_cluster).fit_predict(Z.data.cpu().numpy())

            print(np.unique(_pred))
            _si = SI(Z.data.cpu().numpy()[:5000], _pred[:5000]) if len(np.unique(_pred)) > 1 else 0
            _ch = CH(Z.data.cpu().numpy()[:5000], _pred[:5000]) if len(np.unique(_pred)) > 1 else 0
            cp.plot_all(_pred, save=save, item_file=args.version_name, name='SVAE-CVMFMM', epoch=epoch)
            if save:
                logger = open(os.path.join(self.log_path, "log.txt"), 'a')
                logger.write(
                    "[SVAE-CVMFMM]: epoch: {}, loss: {}, si: {}, ch: {}\n".format(epoch, t_loss, _si, _ch)
                )
                logger.close()
            print("[SVAE-CVMFMM]: epoch: {}, loss: {}, si: {}, ch: {}".format(epoch, t_loss, _si, _ch))


if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-r", "--run_name", dest="run_name", default="SVAE-CvMFMM", help="Name of training run")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=50, type=int, help="Number of epochs")
    parser.add_argument("-s", "--dataset_name", dest="dataset_name", default='adhd', choices=dataset_list,
                        help="Dataset name")
    parser.add_argument("-v", "--version_name", dest="version_name", default="v3")
    args = parser.parse_args()

    input_dim = 176
    group = False
    args.input_dim = 176 if group == False else 857
    args.n_cluster = 40
    func_filenames = [
        '{}/adhd/0010042/0010042_rest_tshift_RPI_voreg_mni.nii.gz'.format(Brain_DIR),
        '{}/adhd/0010064/0010064_rest_tshift_RPI_voreg_mni.nii.gz'.format(Brain_DIR),
        '{}/adhd/0010128/0010128_rest_tshift_RPI_voreg_mni.nii.gz'.format(Brain_DIR),
        # '{}/adhd/0021019/0021019_rest_tshift_RPI_voreg_mni.nii.gz'.format(Brain_DIR),
        # '{}/adhd/0023008/0023008_rest_tshift_RPI_voreg_mni.nii.gz'.format(Brain_DIR),
        # '{}/adhd/0023012/0023012_rest_tshift_RPI_voreg_mni.nii.gz'.format(Brain_DIR),
    ]
    cp = ClusterProcess(model=Trainer(args), n_cluster=args.n_cluster, n_components=input_dim, group=group,
                        sub_num=3, smoothing_fwhm=15., memory="nilearn_cache", threshold=1., memory_level=2,
                        verbose=10, random_state=0)
    cp.fit(func_filenames)
