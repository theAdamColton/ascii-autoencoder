"""
https://github.com/martinarjovsky/WassersteinGAN/blob/master/models/dcgan.py
"""


import torch
import torch.nn as nn
import torch.nn.parallel

import bpdb


class DCGAN_D(nn.Module):
    """
    isize - image size
    nz - size of latent z vector
    nc - number of channels
    ndf - intermediate channel width
    """

    def __init__(self, isize, nc, ndf, ngpu=1, n_extra_layers=0):
        self.name="DCGAN_D"
        super(DCGAN_D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module(
            "initial:{0}-{1}:conv".format(nc, ndf),
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        )
        main.add_module("initial:{0}:relu".format(ndf), nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(
                "extra-layers-{0}:{1}:conv".format(t, cndf),
                nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False),
            )
            main.add_module(
                "extra-layers-{0}:{1}:batchnorm".format(t, cndf), nn.BatchNorm2d(cndf)
            )
            main.add_module(
                "extra-layers-{0}:{1}:relu".format(t, cndf),
                nn.LeakyReLU(0.2, inplace=True),
            )

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module(
                "pyramid:{0}-{1}:conv".format(in_feat, out_feat),
                nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False),
            )
            main.add_module(
                "pyramid:{0}:batchnorm".format(out_feat), nn.BatchNorm2d(out_feat)
            )
            main.add_module(
                "pyramid:{0}:relu".format(out_feat), nn.LeakyReLU(0.2, inplace=True)
            )
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module(
            "final:{0}-{1}:conv".format(cndf, 1),
            nn.Conv2d(cndf, 1, 4, 1, 0, bias=False),
        )
        # Modification
        main.add_module(
            "final:{0}:sigmoid",
            nn.Sigmoid(),
        )
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        # Only one output!
        #output = output.mean(0)
        #return output.view(1)

        # Multiple outputs
        return output

class DLinGan_D(nn.Module):
    def __init__(self, imdim: int, initial_layers = 1, downscale_factor=2, final_layer_size=16):
        self.name = "DeepLinearGanDiscriminator"
        super().__init__()

        assert downscale_factor % 2 == 0
        assert imdim % 2 == 0

        main = nn.Sequential()
        for i in range(initial_layers):
            main.add_module("Lin {}".format(i), nn.Linear(imdim, imdim))
            main.add_module("LRLU {}".format(i), nn.LeakyReLU())
            main.add_module("BN {}".format(i), nn.BatchNorm1d(imdim))

        dim = imdim
        while dim > final_layer_size:
            main.add_module("Lin {}".format(dim), nn.Linear(dim, dim//downscale_factor))
            dim //= downscale_factor
            main.add_module("LRLU {}".format(dim), nn.LeakyReLU())
            main.add_module("BN {}".format(dim), nn.BatchNorm1d(dim))

        main.add_module("Lin out", nn.Linear(dim, 1))
        main.add_module("Sig out", nn.Sigmoid())
        self.main = main

    def forward(self, input):
        return self.main(input)

class DLinGan_G(nn.Module):
    def __init__(self, nz, imdim, initial_layers=1, upscale_factor=2):
        self.name = "DeepLinearGanGenerator"
        super().__init__()

        assert imdim % nz == 0
        assert upscale_factor % 2 == 0

        main = nn.Sequential()
        for i in range(initial_layers):
            main.add_module("Lin {}".format(i), nn.Linear(nz, nz))
            main.add_module("LRLU {}".format(i), nn.LeakyReLU())
            main.add_module("BN {}".format(i), nn.BatchNorm1d(nz))

        dim = nz
        while dim < imdim:
            main.add_module("Lin {}".format(dim), nn.Linear(dim, dim*upscale_factor))
            dim *= upscale_factor
            main.add_module("LRLU {}".format(dim), nn.LeakyReLU())
            main.add_module("BN {}".format(dim), nn.BatchNorm1d(dim))

        main.add_module("Lin out", nn.Linear(dim, imdim))
        main.add_module("LRLU out", nn.LeakyReLU())
        main.add_module("BN out", nn.BatchNorm1d(imdim))

        self.main = main

    def forward(self, input):
        return self.main(input)


class DCGAN_G(nn.Module):
    """
    isize - image size
    nz - size of latent z vector
    nc - number of channels
    ngf - intermediate features
    """

    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        self.name="DCGAN_G"
        super(DCGAN_G, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module(
            "initial:{0}-{1}:convt".format(nz, cngf),
            nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False),
        )
        main.add_module("initial:{0}:batchnorm".format(cngf), nn.BatchNorm2d(cngf))
        main.add_module("initial:{0}:relu".format(cngf), nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize // 2:
            main.add_module(
                "pyramid:{0}-{1}:convt".format(cngf, cngf // 2),
                nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False),
            )
            main.add_module(
                "pyramid:{0}:batchnorm".format(cngf // 2), nn.BatchNorm2d(cngf // 2)
            )
            main.add_module("pyramid:{0}:relu".format(cngf // 2), nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(
                "extra-layers-{0}:{1}:conv".format(t, cngf),
                nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False),
            )
            main.add_module(
                "extra-layers-{0}:{1}:batchnorm".format(t, cngf), nn.BatchNorm2d(cngf)
            )
            main.add_module("extra-layers-{0}:{1}:relu".format(t, cngf), nn.ReLU(True))

        main.add_module(
            "final:{0}-{1}:convt".format(cngf, nc),
            nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False),
        )
        # original
        # main.add_module("final:{0}:tanh".format(nc), nn.Tanh())
        # my change
        main.add_module("final:{0}:sigmoid".format(nc), nn.Sigmoid())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


###############################################################################
class DCGAN_D_nobn(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D_nobn, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        # input is nc x isize x isize
        main.add_module(
            "initial:{0}-{1}:conv".format(nc, ndf),
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        )
        main.add_module("initial:{0}:conv".format(ndf), nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(
                "extra-layers-{0}:{1}:conv".format(t, cndf),
                nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False),
            )
            main.add_module(
                "extra-layers-{0}:{1}:relu".format(t, cndf),
                nn.LeakyReLU(0.2, inplace=True),
            )

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module(
                "pyramid:{0}-{1}:conv".format(in_feat, out_feat),
                nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False),
            )
            main.add_module(
                "pyramid:{0}:relu".format(out_feat), nn.LeakyReLU(0.2, inplace=True)
            )
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module(
            "final:{0}-{1}:conv".format(cndf, 1),
            nn.Conv2d(cndf, 1, 4, 1, 0, bias=False),
        )
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        output = output.mean(0)
        return output.view(1)


class DCGAN_G_nobn(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G_nobn, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module(
            "initial:{0}-{1}:convt".format(nz, cngf),
            nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False),
        )
        main.add_module("initial:{0}:relu".format(cngf), nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize // 2:
            main.add_module(
                "pyramid:{0}-{1}:convt".format(cngf, cngf // 2),
                nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False),
            )
            main.add_module("pyramid:{0}:relu".format(cngf // 2), nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(
                "extra-layers-{0}:{1}:conv".format(t, cngf),
                nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False),
            )
            main.add_module("extra-layers-{0}:{1}:relu".format(t, cngf), nn.ReLU(True))

        main.add_module(
            "final:{0}-{1}:convt".format(cngf, nc),
            nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False),
        )
        main.add_module("final:{0}:tanh".format(nc), nn.Tanh())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
