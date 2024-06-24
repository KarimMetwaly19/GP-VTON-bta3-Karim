from options.train_options import TrainOptions
from models.networks import ResUnetGenerator, load_checkpoint_parallel
import torch.nn as nn
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import cv2
from tqdm import tqdm

modycnt = 1

def show_tryon(i, k):
    combine = torch.cat([i[0], k[0]], 2).squeeze()
    cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
    writer.add_image('combine', (combine.data + 1) / 2.0, step)
    rgb = (cv_img * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite('sample/' + opt.name + '/'+ modycnt +  '.jpg', bgr)
    modycnt = modycnt + 1


#######ResWalid w karim w Mody
class Squeeze_Excitation(nn.Module):
    def __init__(self, channel, r=8):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        b, c, _, _ = inputs.shape
        x = self.pool(inputs).view(b, c)
        x = self.net(x).view(b, c, 1, 1)
        x = inputs * x
        return x

class Stem_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y

class ResNet_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y

class ASPP(nn.Module):
    def __init__(self, in_c, out_c, rate=[1, 6, 12, 18]):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[0], padding=rate[0]),
            nn.BatchNorm2d(out_c)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[1], padding=rate[1]),
            nn.BatchNorm2d(out_c)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[2], padding=rate[2]),
            nn.BatchNorm2d(out_c)
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[3], padding=rate[3]),
            nn.BatchNorm2d(out_c)
        )

        self.c5 = nn.Conv2d(out_c, out_c, kernel_size=1, padding=0)


    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x = x1 + x2 + x3 + x4
        y = self.c5(x)
        return y

class Attention_Block(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        out_c = in_c[1]

        self.g_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[0]),
            nn.ReLU(),
            nn.Conv2d(in_c[0], out_c, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2))
        )

        self.x_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            nn.Conv2d(in_c[1], out_c, kernel_size=3, padding=1),
        )

        self.gc_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

    def forward(self, g, x):
        g_pool = self.g_conv(g)
        x_conv = self.x_conv(x)
        gc_sum = g_pool + x_conv
        gc_conv = self.gc_conv(gc_sum)
        y = gc_conv * x
        return y

class Decoder_Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.a1 = Attention_Block(in_c)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.r1 = ResNet_Block(in_c[0]+in_c[1], out_c, stride=1)

    def forward(self, g, x):
        d = self.a1(g, x)
        d = self.up(d)
        d = torch.cat([d, g], axis=1)
        d = self.r1(d)
        return d
#not working but can
class build_resunetplusplus(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = Stem_Block(36, 64, stride=1)
        self.c2 = ResNet_Block(64, 128, stride=2)
        self.c3 = ResNet_Block(128, 256, stride=2)
        self.c4 = ResNet_Block(256, 512, stride=2)

       #k self.b1 = ASPP(256, 512)
        self.b1 = ASPP(512, 1024)    
        

        self.d1 = Decoder_Block([256, 1024], 512)
        self.d2 = Decoder_Block([128, 512], 256)
        self.d3 = Decoder_Block([64, 256], 128)
        self.output = nn.Conv2d(128, 4, kernel_size=1)
        self.old_lr = opt.lr
        self.old_lr_gmm = 0.1*opt.lr
        

# #habbooda version
    def forward(self, inputs):
        c1 = self.c1(inputs)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)

        b1 = self.b1(c4)

        d1 = self.d1(c3, b1)
        d2 = self.d2(c2, d1)
        d3 = self.d3(c1, d2)

        output = self.output(d3)

        return output

    def update_learning_rate(self, optimizer):
        lrd = opt.lr / opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

print('flag1')

opt = TrainOptions().parse()
os.makedirs('sample/'+opt.name,exist_ok=True)
print('flag2')
def CreateDataset(opt):
    if opt.dataset == 'vitonhd':
        from data.aligned_dataset_vitonhd import AlignedDataset
        dataset = AlignedDataset()
        dataset.initialize(opt, mode='test')
    elif opt.dataset == 'dresscode':
        from data.aligned_dataset_dresscode import AlignedDataset
        dataset = AlignedDataset()
        dataset.initialize(opt, mode='test', stage='gen')
    return dataset

torch.cuda.set_device(opt.local_rank)
torch.distributed.init_process_group(
    'nccl',
    init_method='env://'
)
device = torch.device(f'cuda:{opt.local_rank}')
print('flag3')

train_data = CreateDataset(opt)
train_sampler = DistributedSampler(train_data)
train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=False,
                          num_workers=4, pin_memory=True, sampler=train_sampler)

# gen_model = ResUnetGenerator(36, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
gen_model = build_resunetplusplus()
# gen_model.train()
gen_model.eval()
gen_model.cuda()

print(opt.PBAFN_gen_checkpoint + 'mody&karim are very happy')
load_checkpoint_parallel(gen_model, opt.PBAFN_gen_checkpoint)

# gen_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(gen_model).to(device)
if opt.isTrain and len(opt.gpu_ids):
    model_gen = torch.nn.parallel.DistributedDataParallel(gen_model, device_ids=[opt.local_rank])
else:
    model_gen = gen_model

for data in tqdm(train_loader):
    real_image = data['image'].cuda()
    clothes = data['color'].cuda()
    preserve_mask = data['preserve_mask3'].cuda()
    preserve_region = real_image * preserve_mask
    warped_cloth = data['warped_cloth'].cuda()
    warped_prod_edge = data['warped_edge'].cuda()
    arms_color = data['arms_color'].cuda()
    arms_neck_label= data['arms_neck_lable'].cuda()
    pose = data['pose'].cuda()

    gen_inputs = torch.cat([preserve_region, warped_cloth, warped_prod_edge, arms_neck_label, arms_color, pose], 1)

    gen_outputs = model_gen(gen_inputs)
    p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
    p_rendered = torch.tanh(p_rendered)
    m_composite = torch.sigmoid(m_composite)
    m_composite = m_composite * warped_prod_edge
    p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
    k = p_tryon
    i = p_rendered



    show_tryon(i, k)

    
    
    bz = pose.size(0)
    for bb in range(bz):
        combine = k[bb].squeeze()
    
        cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy()+1)/2
        rgb = (cv_img*255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        cloth_id = data['color_path'][bb].split('/')[-1]
        person_id = data['img_path'][bb].split('/')[-1]
        c_type = data['c_type'][bb]
        save_path = 'sample/'+opt.name+'/'+c_type+'___'+person_id+'___'+cloth_id[:-4]+'.png'
        cv2.imwrite(save_path, bgr)
