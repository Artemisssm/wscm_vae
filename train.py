import os
from config import *
import utils
from wvae import *
from torch import optim
from tqdm import tqdm
from utils import *
from torchvision.utils import save_image
from torch.nn import functional as F
import matplotlib.pyplot as plt

import seaborn as sns

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda')
args = get_config()

if 'pendulum' in args.dataset:
    label_idx = range(4)
elif 'human' in args.dataset:
    label_idx = range(6)
elif 'tree' in args.dataset:
    label_idx = range(7)
else:
    if args.labels == 'smile':
        label_idx = [31, 20, 19, 21, 23, 13]
    elif args.labels == 'age':
        label_idx = [39, 20, 28, 18, 13, 3]
    else:
        raise NotImplementedError("Not supported structure.")
num_label = len(label_idx)

save_dir = './results/{}/{}_{}_sup{}_seed{}/'.format(
    args.dataset, args.labels, args.prior, str(args.sup_type), args.seed)
utils.make_folder(save_dir)
utils.write_config_to_file(args, save_dir)

celoss = torch.nn.BCEWithLogitsLoss()
# adversarial_loss = torch.nn.BCELoss()

# 得到数据集
train_loader, test_loader, train_set = utils.make_dataloader(args)
log_file_name = os.path.join(save_dir, 'log.txt')

if args.resume:
    log_file = open(log_file_name, "at")
else:
    log_file = open(log_file_name, "wt")

if 'scm' in args.prior:
    A = torch.zeros((num_label, num_label))
    if args.labels == 'smile':
        A[0, 2:6] = 1
        A[1, 4] = 1
    elif args.labels == 'age':
        A[0, 2:6] = 1
        A[1, 2:4] = 1
    elif args.labels == 'pend':
        A[0, 2:4] = 1
        A[1, 2:4] = 1
    elif args.labels == 'hum':
        A[0, 3:6] = 1
        A[1, 2:6] = 1
    elif args.labels == 'tre':
        A[0, 4:6] = 1
        A[1, 2:4] = 1
        A[3, 4:7] = 1
else:
    A = None

print('Build models...')
model = WVAE(args.latent_dim, args.g_conv_dim, args.image_size,
             args.enc_dist, args.enc_arch, args.enc_fc_size, args.enc_noise_dim, args.dec_dist,
             args.prior, num_label, A, args.reconstruction_loss)

discriminator = BigJointDiscriminator(args.latent_dim, args.d_conv_dim, args.image_size,
                                          args.dis_fc_size)



A_optimizer = None
prior_optimizer = None
if 'scm' in args.prior:
    enc_param = model.encoder.parameters()  # 获取模型的编码器的参数，并赋值给enc_param
    dec_param = model.decoder.parameters()  # 获取模型的解码器的参数，并转换成列表赋值给dec_param
    prior_param = list(model.prior.parameters())  # 获取模型的先验的参数，并转换成列表赋值给prior_param
    A_optimizer = optim.Adam(prior_param[0:1], lr=args.lr_a)  # 创建一个Adam优化器，用来优化prior_param中的第一个参数（即A），并赋值给A_optimizer
    prior_optimizer = optim.Adam(prior_param[1:], lr=args.lr_p, betas=(
        args.beta1, args.beta2))  # 创建一个Adam优化器，用来优化prior_param中的其余参数，并赋值给prior_optimizer
else:
    enc_param = model.encoder.parameters()
    dec_param = model.decoder.parameters()

encoder_optimizer = optim.Adam(enc_param, lr=args.lr_e,
                               betas=(args.beta1, args.beta2))  # 创建一个Adam优化器，用来优化enc_param中的参数，并赋值给encoder_optimizer
decoder_optimizer = optim.Adam(dec_param, lr=args.lr_g,
                               betas=(args.beta1, args.beta2))  # 创建一个Adam优化器，用来优化dec_param中的参数，并赋值给decoder_optimizer

D_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2)) # 创建一个Adam优化器，用来优化discriminator中的参数，并赋值给D_optimizer


model = nn.DataParallel(model.to(device))
discriminator = nn.DataParallel(discriminator.to(device))


def test(epoch, i, model, test_data, save_dir, fixed_noise, fixed_zeros):
    model.eval()
    with torch.no_grad():  # 不计算梯度
        x = test_data.to(device)  # 把测试数据转移到device上，并赋值给x

        # Reconstruction
        x_recon = model(x, recon=True)  # 调用模型，得到重建后的图像，并赋值给x_recon
        recons = utils.draw_recon(x.cpu(), x_recon.cpu())  # 调用utils.draw_recon函数，绘制原始图像和重建图像的对比，并赋值给recons
        del x_recon  # 删除x_recon，释放内存
        save_image(recons, save_dir + 'recon_' + str(epoch) + '_' + str(i) + '.png', nrow=args.nrow,normalize=True, scale_each=True)  # 保存recons到指定的路径，使用args.nrow指定每行的图片数量，使用normalize和scale_each进行归一化

        # Generation
        sample = model(z=fixed_noise, gen=True).cpu()  # 调用模型，得到生成的图像，并转移到cpu上，并赋值给sample
        save_image(sample, save_dir + 'gen_' + str(epoch) + '_' + str(i) + '.png', normalize=True,scale_each=True)  # 保存sample到指定的路径，使用normalize和scale_each进行归一化

        # Traversal (given a fixed traversal range)
        sample = model.module.traverse(fixed_zeros).cpu()  # 调用model.module.traverse函数，得到遍历隐变量后的图像，并转移到cpu上，并赋值给sample
        save_image(sample, save_dir + 'trav_' + str(epoch) + '_' + str(i) + '.png', normalize=True, scale_each=True, nrow=10)  # 保存sample到指定的路径，使用normalize和scale_each进行归一化，使用nrow=10指定每行的图片数量
        del sample  # 删除sample，释放内存

    model.train()


if args.prior == 'uniform':  # 如果先验分布是均匀分布
    fixed_noise = torch.rand(args.save_n_samples, args.latent_dim,
                             device=device) * 2 - 1  # 生成一个服从[-1,1]区间的随机张量，并赋值给fixed_noise
else:  # 否则
    fixed_noise = torch.randn(args.save_n_samples, args.latent_dim,
                              device=device)  # 生成一个服从标准正态分布的随机张量，并赋值给fixed_noise
fixed_unif_noise = torch.rand(1, args.latent_dim, device=device) * 2 - 1  # 生成一个服从[-1,1]区间的随机张量，并赋值给fixed_unif_noise
fixed_zeros = torch.zeros(1, args.latent_dim, device=device)  # 生成一个全零张量，并赋值给fixed_zeros



print('Start training...')
for epoch in range(args.start_epoch, args.start_epoch + args.n_epochs):
    pbar = tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch}')

    model.train()

    for batch_idx, (x, label) in enumerate(pbar):
        x = x.to(device)
        sup_flag = label[:, 0] != -1  # 判断label的第一列是否不等于-1，得到一个布尔张量，并赋值给sup_flag
        if sup_flag.sum() > 0:  # 如果sup_flag中为True的元素个数大于0，说明有有效的标签
            label = label[sup_flag, :][:, label_idx].float()  # 用sup_flag筛选出有效的标签，并用label_idx选择需要的列，然后转换成浮点类型，并赋值给label
        if 'pendulum' or 'tree' in args.dataset:  # 如果args.dataset中包含'pendulum'或者'human'
            if args.sup_type == 'ce':  # 如果args.sup_type是'ce'
                # Normalize labels to 0,1
                scale = get_scale(train_set)  # 调用get_scale函数，获取训练集的最小值和最大值，并赋值给scale
                label = (label - scale[0]) / (scale[1] - scale[0])  # 用scale对label进行归一化，使其范围在0到1之间，并赋值给label
            else:  # 否则
                # Normalize labels to mean 0 std 1
                mm, ss = get_stats()  # 调用get_stats函数，获取训练集的均值和标准差，并赋值给mm和ss
                label = (label - mm) / ss  # 用mm和ss对label进行标准化，使其均值为0，标准差为1，并赋值给label
            num_labels = len(label_idx)  # 获取label_idx的长度，并赋值给num_labels
            label = label.to(device)  # 把label转移到device上，并赋值给label


        for _ in range(args.d_steps_per_iter): # 循环args.d_steps_per_iter次
            discriminator.zero_grad() # 把判别器的梯度清零

            # Sample z from prior p_z
            if args.prior == 'uniform': # 如果先验分布是均匀分布
                z = torch.rand(x.size(0), args.latent_dim, device=x.device) * 2 - 1 # 生成一个服从[-1,1]区间的随机张量，并赋值给z
            else: # 否则
                z = torch.randn(x.size(0), args.latent_dim, device=x.device) # 生成一个服从标准正态分布的随机张量，并赋值给z

            # Get inferred latent z = E(x) and generated image x = G(z)
            if 'scm' in args.prior: # 如果args.prior中包含'scm'
                x_fake, z_fake, z_mu, z_logvar = model(x) # 调用模型，得到编码后的隐变量z_fake，生成后的图像x_fake，真实的隐变量z和其他输出，并赋值给相应的变量
            else: # 否则
                x_fake, z_fake, z_mu, z_logvar = model(x) # 调用模型，得到编码后的隐变量z_fake，生成后的图像x_fake和其他输出，并赋值给相应的变量

            # Compute D loss
            x_score = discriminator(x) # 调用判别器，得到编码后的隐变量z_fake对应的分数，并赋值给encoder_score
            x_fake_score = discriminator(x_fake.detach()) # 调用判别器，得到生成后的图像x_fake对应的分数，并赋值给decoder_score

            one = torch.full((x.size(0),), 1., device=x.device)
            zero = torch.full((x.size(0),), 0., device=x.device)

            loss_d_x = celoss(x_score, one)
            loss_d_x_fake = celoss(x_fake_score, zero)

            # z_fake_s = discriminator(x_fake.detach(), z_fake.detach())
            # z_s = discriminator(x_fake.detach(), z)
            # recon_loss = F.softplus(z_s).mean() + F.softplus(-z_fake_s).mean()
            # recon_loss = celoss(z_fake_s, z_s)

            loss_d = loss_d_x + loss_d_x_fake

            # encoder_score = discriminator(x_fake.detach(), z_fake.detach())
            # decoder_score = discriminator(x_fake.detach(), z.detach())
            # loss_d_three = F.softplus(decoder_score).mean() + F.softplus(-encoder_score).mean()

            loss_d.backward()

            D_optimizer.step() # 调用D_optimizer，更新判别器的参数



        # train model
        # model.zero_grad()
        for _ in range(args.g_steps_per_iter):

            if args.prior == 'uniform':  # 如果先验分布是均匀分布
                z = torch.rand(x.size(0), args.latent_dim, device=x.device) * 2 - 1  # 生成一个服从[-1,1]区间的随机张量，并赋值给z
            else:  # 否则
                z = torch.randn(x.size(0), args.latent_dim, device=x.device)  # 生成一个服从标准正态分布的随机张量，并赋值给z

                # Get inferred latent z = E(x) and generated image x = G(z)
            if 'scm' in args.prior:  # 如果args.prior中包含'scm'
                x_fake, z_fake, z_mu, z_logvar = model(x)  # 调用模型，得到编码后的隐变量z_fake，生成后的图像x_fake，真实的隐变量z和其他输出，并赋值给相应的变量
            else:  # 否则
                x_fake, z_fake, z_mu, z_logvar = model(x)  # 调用模型，得到编码后的隐变量z_fake，生成后的图像x_fake和其他输出，并赋值给相应的变量

            model.zero_grad()
            if sup_flag.sum() > 0:  # 如果sup_flag中为True的元素个数大于0，说明有有效的标签
                label_z = z_mu[sup_flag, :num_labels]  # 用sup_flag筛选出有效的隐变量，并用num_labels选择需要的列，并赋值给label_z
                if 'pendulum' or 'tree' in args.dataset:  # 如果args.dataset中包含'pendulum'
                    if args.sup_type == 'ce':  # 如果args.sup_type是'ce'
                        # CE loss
                        sup_loss = celoss(label_z, label)  # 计算label_z和label之间的交叉熵损失，并赋值给sup_loss
                    else:  # 否则
                        # l2 loss
                        sup_loss = nn.MSELoss()(label_z, label)  # 计算label_z和label之间的均方误差损失，并赋值给sup_loss
                else:  # 否则
                    sup_loss = celoss(label_z, label)  # 计算label_z和label之间的交叉熵损失，并赋值给sup_loss，label是经过先验之后得到的标签，label_z是刚得到的标签
            else:  # 否则
                sup_loss = torch.zeros([1], device=device)  # 生成一个全零张量，并赋值给sup_loss

            # KLD = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=-1)  # 调用判别器，得到编码后的隐变量z_fake对应的分数，并赋值给encoder_score

            # z_fake_s = discriminator(x_fake, z_fake)
            loss_r, recon_loss, KLD = model.module.loss_function(x_fake, x, z_mu, z_logvar, x.shape[0])
            loss = loss_r + celoss(discriminator(x_fake), one) + sup_loss * args.sup_coef
            # r_encoder = torch.exp(z_fake_s.detach())  # 对decoder_score进行detach操作，然后取指数，并赋值给r_decoder
            # s_encoder = r_encoder.clamp(0.5, 2)  # 对r_decoder进行截断操作，使其范围在0.5到2之间，并赋值给s_decoder
            # z_fake_s_s = (s_encoder * z_fake_s).mean()  # 计算解码器的损失函数，使用s_decoder和decoder_score的乘积的负平均值，并赋值给loss_decoder
            # loss_encoder = encoder_score.mean()  # 计算encoder_score的平均值，并赋值给loss_encoder
            # loss, recon_loss, kld = model.module.loss_function(x_fake, x, z_fake_mean, z_fake_logvar, z_fake)
            # loss_encoder = sup_loss * args.sup_coef + z_fake_s.mean()
            # decoder_score = discriminator(x_fake, z)  # 调用判别器，得到生成后的图像x_fake对应的分数，并赋值给decoder_score
            # with scaling clipping for stabilization
            # r_decoder = torch.exp(decoder_score.detach())  # 对decoder_score进行detach操作，然后取指数，并赋值给r_decoder
            # s_decoder = r_decoder.clamp(0.5, 2)  # 对r_decoder进行截断操作，使其范围在0.5到2之间，并赋值给s_decoder
            #
            # loss_decoder = -(s_decoder * decoder_score).mean()  # 计算解码器的损失函数，使用s_decoder和decoder_score的乘积的负平均值，并赋值给loss_decoder



            # z_s = discriminator(x_fake, z)
            # r_encoder = torch.exp(z_s.detach())  # 对decoder_score进行detach操作，然后取指数，并赋值给r_decoder
            # s_encoder = r_encoder.clamp(0.5, 2)  # 对r_decoder进行截断操作，使其范围在0.5到2之间，并赋值给s_decoder
            # z_s_s = -(s_encoder * z_s).mean()
            loss.backward()  # 对loss_decoder进行反向传播，计算梯度

            encoder_optimizer.step()
            decoder_optimizer.step()  # 调用decoder_optimizer，更新解码器的参数
            if 'scm' in args.prior:  # 如果args.prior中包含'scm'
                model.module.prior.set_zero_grad()
                A_optimizer.step()  # 调用A_optimizer，更新因果层的参数
                prior_optimizer.step()  # 调用prior_optimizer，更新先验网络的参数

            # loss_encoder = sup_loss * args.sup_coef  # 把loss_encoder和sup_loss乘以args.sup_coef的结果相加，并赋值给loss_encoder
            #
            # loss_decoder = model.module.loss_function(x_fake, x, z_fake, z_fake_logvar, z)['loss']
            # loss = loss_encoder + loss_decoder
            # model.module.zero_grad()
            # if epoch == 1:
            #     if batch_idx != 0:
            #         model.module.prior.set_zero_grad()
            # else:
            #     model.module.prior.set_zero_grad()
            # loss.backward()
            # encoder_optimizer.step()
            # decoder_optimizer.step()
            # if 'scm' in args.prior:  # 如果args.prior中包含'scm'
            #     A_optimizer.step()  # 调用A_optimizer，更新因果层的参数
            #     prior_optimizer.step()  # 调用prior_optimizer，更新先验网络的参数

            if batch_idx == 0 or (batch_idx + 1) % args.print_every == 0:
                log = (
                    'Train Epoch: {} ({:.0f}%)\t, loss_d: {:.4f}, recon_loss:{:.4f}, KLD:{:.4f}, Sup loss: {:.4f}, loss: {:.4f}'.format(
                        epoch, 100. * batch_idx / len(train_loader),
                        loss_d.item(), recon_loss.item(), KLD.item(), sup_loss.item(), loss.item()))
                print(log)
                log_file.write(log + '\n')
                log_file.flush()

            if (epoch == 1 or epoch % args.sample_every_epoch == 0) and batch_idx == len(train_loader) - 1:  # 如果epoch是1或者能被args.sample_every_epoch整除，并且batch_idx是最后一个批次的索引
                test(epoch, batch_idx + 1, model, x[:args.save_n_recons], save_dir, fixed_noise, fixed_zeros)  # 调用test函数，传入epoch, batch_idx + 1, model, x的前args.save_n_recons

            if (epoch == 1 or epoch % 10 == 0) and batch_idx == len(train_loader) - 1:
                fig_name = 'prior_A_{}'.format(epoch)
                fig_path = save_dir + '/' + fig_name
                plt.figure()
                fig = sns.heatmap(prior_param[0:1][0].data.cpu().numpy(), annot=True, cmap='Blues')
                heatmap = fig.get_figure()
                heatmap.savefig(fig_path, dpi=400)

            if (epoch % 10 == 0) and batch_idx == len(train_loader) - 1:
                torch.save(model.state_dict(), save_dir + "model_" + str(epoch) + ".pth")


