from GAN.model import generator, discriminator
import argparse
import os
import random
import torch
import torch.nn as nn
import pathlib
from skimage import io
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from inpaint_tools import read_file_list
from tqdm import tqdm
from inpaint_config import InPaintConfig
import numpy

epochs = 100
Batch_Size = 64
lr = 0.0002
beta1 = 0.5
over = 4

args = argparse.ArgumentParser(description='InpaintImages')
config = InPaintConfig(args)
settings = config.settings

ngpu = 1
wtl2 = 0.999

# custom weights initialization called on netG and netD


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


resume_epoch = 0

netG = generator()
netG.apply(weights_init)


netD = discriminator()
netD.apply(weights_init)

criterion = nn.BCELoss()
criterionMSE = nn.MSELoss()

input_real = torch.FloatTensor(Batch_Size, 3, 360, 360)
input_masked = torch.FloatTensor(Batch_Size, 3, 360, 360)
label = torch.FloatTensor(Batch_Size)
real_label = 1
fake_label = 0

real_center = torch.FloatTensor(Batch_Size, 3, 64, 64)


netD.cuda()
netG.cuda()
criterion.cuda()
criterionMSE.cuda()
input_real, input_masked, label = input_real.cuda(
), input_masked.cuda(), label.cuda()
real_center = real_center.cuda()


input_real = Variable(input_real)
input_masked = Variable(input_masked)
label = Variable(label)


real_center = Variable(real_center)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))



for epoch in range(resume_epoch, epochs):
    input_data_dir = settings["dirs"]["input_data_dir"]
    output_data_dir = settings["dirs"]["output_data_dir"]
    data_set = settings["data_set"]
    model_dir = os.path.join(output_data_dir, "trained_model")

    inpainted_result_dir = os.path.join(output_data_dir, f"inpainted_{data_set}")
    pathlib.Path(inpainted_result_dir).mkdir(parents=True, exist_ok=True)

    print(f"InPainting {data_set} and placing results in {inpainted_result_dir} with model from {model_dir}")

    avg_img_name = os.path.join(model_dir, "average_image.png")
    avg_img = io.imread(avg_img_name)

    file_list = os.path.join(input_data_dir, "data_splits", data_set + ".txt")
    file_ids = read_file_list(file_list)

    print(f"Inpainting {len(file_ids)} images")

    for idx in tqdm(file_ids):
        # in_masked_image = os.path.join(input_data_dir, "masked", f"{idx}_stroke_masked.png")
        in_mask_image = os.path.join(input_data_dir, "masks", f"{idx}_stroke_mask.png")
        in_original_image = os.path.join(input_data_dir, "originals", f"{idx}.jpg")

        out_image_name = os.path.join(inpainted_result_dir, f"{idx}.png")

        # real_cpu = torch.from_numpy(io.imread(in_original_image))
        original_image_matrix = numpy.matrix(io.imread(in_original_image))
        mask = torch.from_numpy(io.imread(in_mask_image))

        mask_indexes = [(index, row.index(255)) for index, row in enumerate(original_image_matrix) if 255 in row]

        input_real = torch.from_numpy(io.imread(in_original_image))
        input_masked = input_real[mask_indexes] = mask[mask_indexes]
        batch_size = input_real.size(0)

        with torch.no_grad():
            input_real.resize_(input_real.size()).copy_(input_real)
            input_masked.resize_(input_masked.size()).copy_(input_masked)

        #start the discriminator by training with real data---
        netD.zero_grad()
        with torch.no_grad():
            label.resize_(batch_size).fill_(real_label)

        output = netD(real_center)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        # train the discriminator with fake data---
        fake = netG(input_masked)
        label.data.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        # train the generator now---
        netG.zero_grad()
        label.data.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG_D = criterion(output, label)

        wtl2Matrix = real_center.clone()
        wtl2Matrix.data.fill_(wtl2*10)
        wtl2Matrix.data[:, :, int(over):int(
            360/2 - over), int(over):int(360/2 - over)] = wtl2


        errG = (1-wtl2) * errG_D + wtl2

        errG.backward()

        D_G_z2 = output.data.mean()
        optimizerG.step()

        print('[%d / %d][%d / %d] Loss_D: %.4f Loss_G: %.4f / %.4f l_D(x): %.4f l_D(G(z)): %.4f'
            % (epoch, epochs, idx, len(file_ids)),
                errD.data, errG_D.data, D_x, D_G_z1, )

        if idx % 100 == 0:
        
            vutils.save_image(real_cpu, out_image_name)
            vutils.save_image(input_masked.data, out_image_name)
            recon_image = input_masked.clone()
            recon_image.data[:, :, int(
                360/4):int(360/4+360/2), int(360/4):int(360/4+360/2)] = fake.data
            vutils.save_image(recon_image.data, out_image_name)
