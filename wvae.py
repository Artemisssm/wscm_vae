import torch.nn as nn
from resnet import *
import torch
from sagan import *
from causal_model import *
import math


class Encoder(nn.Module):
    r'''ResNet Encoder

    Args:
        latent_dim: latent dimension
        arch: network architecture. Choices: resnet - resnet50, resnet18
        dist: encoder distribution. Choices: deterministic, gaussian, implicit
        fc_size: number of nodes in each fc layer
        noise_dim: dimension of input noise when an implicit encoder is used
    '''

    def __init__(self, latent_dim=64, arch='resnet', dist='gaussian', fc_size=2048, noise_dim=128):
        super().__init__()  # 调用父类的初始化函数
        self.latent_dim = latent_dim  # 把隐变量的维度赋值给self.latent_dim
        self.dist = dist  # 把隐变量的分布赋值给self.dist
        self.noise_dim = noise_dim  # 把噪声的维度赋值给self.noise_dim

        in_channels = noise_dim + 3 if dist == 'implicit' else 3  # 根据dist的值，确定输入通道的数量
        out_dim = latent_dim * 2 if dist == 'gaussian' else latent_dim  # 根据dist的值，确定输出维度的大小
        if arch == 'resnet':  # 如果架构是'resnet'
            self.encoder = resnet50(pretrained=False, in_channels=in_channels, fc_size=fc_size,
                                    out_dim=out_dim)  # 创建一个预训练的resnet50编码器，并把它赋值给self.encoder
        else:  # 否则
            assert arch == 'resnet18'  # 断言架构是'resnet18'
            self.encoder = resnet18(pretrained=False, in_channels=in_channels, fc_size=fc_size,
                                    out_dim=out_dim)  # 创建一个预训练的resnet18编码器，并把它赋值给self.encoder

    def forward(self, x, avepool=False):
        '''
        :param x: input image
        :param avepool: whether to return the average pooling feature (used for downstream tasks)
        :return:
        '''
        if self.dist == 'implicit':  # 如果self.dist是'implicit'
            # Concatenate noise with the input image x
            noise = x.new(x.size(0), self.noise_dim, 1, 1).normal_(0, 1)  # 生成一个服从标准正态分布的随机张量，并赋值给noise
            noise = noise.expand(x.size(0), self.noise_dim, x.size(2), x.size(3))  # 把noise扩展到和x相同的维度，并赋值给noise
            x = torch.cat([x, noise], dim=1)  # 把x和noise在第二个维度上拼接起来，并赋值给x
        z, ap = self.encoder(x)  # 调用self.encoder，得到隐变量z和平均池化后的输出ap，并赋值给相应的变量
        if avepool:  # 如果avepool为真
            return ap  # 返回ap
        if self.dist == 'gaussian':  # 如果self.dist是'gaussian'
            return z.chunk(2, dim=1)  # 把z在第二个维度上分成两个张量，并返回
        else:  # 否则
            return z  # 返回z


# encoder = Encoder()
# # print((encoder))
# x = torch.rand(16, 3, 64, 64)
# z_mu, z_a = encoder(x)
# print(z_mu, z_a)


class Decoder(nn.Module):
    r'''Big generator based on SAGAN

    Args:
        latent_dim: latent dimension
        conv_dim: base number of channels
        image_size: image resolution
        dist: generator distribution. Choices: deterministic, gaussian, implicit
        g_std: scaling the standard deviation of the gaussian generator. Default: 1
    '''

    def __init__(self, latent_dim=64, conv_dim=32, image_size=64, dist='deterministic', g_std=1):
        super().__init__()
        self.latent_dim = latent_dim  # 把隐变量的维度赋值给self.latent_dim
        self.dist = dist  # 把隐变量的分布赋值给self.dist
        self.g_std = g_std  # 把高斯分布的标准差赋值给self.g_std

        out_channels = 6 if dist == 'gaussian' else 3  # 根据dist的值，确定输出通道的数量
        add_noise = True if dist == 'implicit' else False  # 根据dist的值，确定是否在解码器中添加噪声
        self.decoder = Generator(latent_dim, conv_dim, image_size, out_channels,
                                 add_noise)  # 创建一个生成器，并把它赋值给self.decoder

    def forward(self, z, mean=False, stats=False):
        out = self.decoder(z)  # 调用self.decoder，得到输出，并赋值给out
        if self.dist == 'gaussian':  # 如果self.dist是'gaussian'
            x_mu, x_logvar = out.chunk(2, dim=1)  # 把out在第二个维度上分成两个张量，并赋值给x_mu和x_logvar
            if stats:  # 如果stats为真
                return x_mu, x_logvar  # 返回x_mu和x_logvar
            else:  # 否则
                x_sample = reparameterize(x_mu, (x_logvar / 2).exp(),
                                          self.g_std)  # 调用reparameterize函数，根据x_mu和x_logvar生成样本，并赋值给x_sample
                if mean:  # 如果mean为真
                    return x_mu  # 返回x_mu
                else:  # 否则
                    return x_sample  # 返回x_sample
        else:  # 否则
            return out  # 返回out


# decoder = Decoder()
# z = torch.rand(16, 64)
# x = decoder(z)
# print(x)


def reparameterize(mu, sigma, std=1):
    assert mu.shape == sigma.shape
    eps = mu.new(mu.shape).normal_(0, std)
    return mu + sigma * eps


class WVAE(nn.Module):
    num_iter = 0

    def __init__(self, latent_dim=64, conv_dim=32, image_size=64,
                 enc_dist='gaussian', enc_arch='resnet', enc_fc_size=2048, enc_noise_dim=128, dec_dist='implicit',
                 prior='gaussian', num_label=None, A=None, anneal_steps=200, alpha=1, beta=6, gamma=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.enc_dist = enc_dist
        self.dec_dist = dec_dist
        self.prior_dist = prior
        self.num_label = num_label
        self.anneal_steps = anneal_steps
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.encoder = Encoder(latent_dim, enc_arch, enc_dist, enc_fc_size, enc_noise_dim)
        self.decoder = Decoder(latent_dim, conv_dim, image_size, dec_dist)
        if 'scm' in prior:
            self.prior = SCM(num_label, A, scm_type=prior)

    def encode(self, x, mean=False, avepool=False):
        if avepool:
            return self.encoder(x, avepool=True)
        else:
            if self.enc_dist == 'gaussian':
                z_mu, z_logvar = self.encoder(x)
                if mean:  # for downstream tasks
                    return z_mu
                else:
                    z_fake = reparameterize(z_mu, (z_logvar / 2).exp())
                    return z_fake
            else:
                return self.encoder(x)

    def decode(self, z, mean=True):
        if self.decoder_type != 'gaussian':
            return self.decoder(z)
        else:  # gaussian
            return self.decoder(z, mu=mean)

    def traverse(self, eps, gap=3, n=10):
        dim = self.num_label if self.num_label is not None else self.latent_dim  # 如果self.num_label不是None，就把它赋值给dim，否则就把self.latent_dim赋值给dim
        sample = torch.zeros((n * dim, 3, self.image_size, self.image_size))  # 生成一个全零张量，并赋值给sample
        eps = eps.expand(n, self.latent_dim)  # 把eps扩展到(n, self.latent_dim)的维度，并赋值给eps
        if self.prior_dist == 'gaussian' or self.prior_dist == 'uniform':  # 如果self.prior_dist是'gaussian'或者'uniform'
            z = eps  # 把eps赋值给z
        else:  # 否则
            label_z = self.prior(eps[:, :dim])  # 调用self.prior，得到eps的前dim个维度经过因果层后的输出，并赋值给label_z
            other_z = eps[:, dim:]  # 得到eps的剩余维度，并赋值给other_z
            z = torch.cat([label_z, other_z], dim=1)  # 把label_z和other_z在第二个维度上拼接起来，并赋值给z
        for idx in range(dim):  # 循环dim次
            traversals = torch.linspace(-gap, gap, steps=n)  # 生成一个在[-gap, gap]区间上均匀分布的张量，并赋值给traversals
            z_new = z.clone()  # 复制z，并赋值给z_new
            z_new[:, idx] = traversals  # 把traversals赋值给z_new的第idx个维度
            with torch.no_grad():  # 不计算梯度
                sample[n * idx:(n * (idx + 1)), :, :, :] = self.decoder(z_new)  # 调用self.decoder，得到生成的图像，并赋值给sample的相应位置
        return sample  # 返回sample

    def forward(self, x=None, z=None, recon=False, infer_mean=True):
        # recon_mean is used for gaussian decoder which we do not use here.
        # Training Mode
        if x is not None and z is None:  # 如果x和z都不是None
            if self.enc_dist == 'gaussian':  # 如果self.enc_dist是'gaussian'
                z_mu, z_logvar = self.encoder(x)  # 调用self.encoder，得到隐变量的均值和对数方差，并赋值给z_mu和z_logvar
            else:  # deterministic or implicit
                z_fake = self.encoder(x)  # 调用self.encoder，得到隐变量，并赋值给z_fake


            if 'scm' in self.prior_dist:  # 如果self.prior_dist中包含'scm'
                # in prior
                label_z = self.prior(z_mu[:, :self.num_label])  # 调用self.prior，得到z的前self.num_label个维度经过因果层后的输出，并赋值给label_z
                other_z = z_mu[:, self.num_label:]  # 得到z的剩余维度，并赋值给other_z
                z_fake = torch.cat([label_z, other_z], dim=1)  # 把label_z和other_z在第二个维度上拼接起来，并赋值给z

            z = reparameterize(z_fake, (z_logvar / 2).exp())

            x_fake = self.decoder(z)  # 调用self.decoder，得到生成的图像，并赋值给x_fake

            if recon == True:
                return x_fake

            if 'scm' in self.prior_dist:  # 如果self.prior_dist中包含'scm'
                if self.enc_dist == 'gaussian' and infer_mean:  # 如果self.enc_dist是'gaussian'并且infer_mean为真
                    return z_fake, x_fake, z, z_mu, z_logvar  # 返回z_fake, x_fake, z, z_mu
                else:  # 否则
                    return z_fake, x_fake, z, None, z_logvar  # 返回z_fake, x_fake, z, None
            return z_fake, x_fake, z_mu, z_logvar  # 返回z_fake, x_fake, z_mu



        # Generation Mode
        elif x is None and z is not None:  # 如果x是None而z不是None
            # if 'scm' in self.prior_dist:  # 如果self.prior_dist中包含'scm'
            #     label_z = self.prior(z[:, :self.num_label])  # 调用self.prior，得到z的前self.num_label个维度经过因果层后的输出，并赋值给label_z
            #     other_z = z[:, self.num_label:]  # 得到z的剩余维度，并赋值给other_z
            #     z = torch.cat([label_z, other_z], dim=1)  # 把label_z和other_z在第二个维度上拼接起来，并赋值给z
            return self.decoder(z)  # 返回调用self.decoder后的结果

    def log_density_gaussian(self, x, mu, logvar):
        """
        Computes the log pdf of the Gaussian with parameters mu and logvar at x
        :param x: (Tensor) Point at whichGaussian PDF is to be evaluated
        :param mu: (Tensor) Mean of the Gaussian distribution
        :param logvar: (Tensor) Log variance of the Gaussian distribution
        :return:
        """
        norm = - 0.5 * (math.log(2 * math.pi) + logvar)
        log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
        return log_density

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        z = args[4]

        weight = 1  # kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input, reduction='sum')

        log_q_zx = self.log_density_gaussian(z, mu, log_var).sum(dim=1)

        zeros = torch.zeros_like(z)
        log_p_z = self.log_density_gaussian(z, zeros, zeros).sum(dim=1)

        batch_size, latent_dim = z.shape
        mat_log_q_z = self.log_density_gaussian(z.view(batch_size, 1, latent_dim),
                                                mu.view(1, batch_size, latent_dim),
                                                log_var.view(1, batch_size, latent_dim))

        # Reference
        # [1] https://github.com/YannDubs/disentangling-vae/blob/535bbd2e9aeb5a200663a4f82f1d34e084c4ba8d/disvae/utils/math.py#L54
        dataset_size = (1 / weight) * batch_size  # dataset size
        strat_weight = (dataset_size - batch_size + 1) / (dataset_size * (batch_size - 1))
        importance_weights = torch.Tensor(batch_size, batch_size).fill_(1 / (batch_size - 1)).to(input.device)
        importance_weights.view(-1)[::batch_size] = 1 / dataset_size
        importance_weights.view(-1)[1::batch_size] = strat_weight
        importance_weights[batch_size - 2, 0] = strat_weight
        log_importance_weights = importance_weights.log()

        mat_log_q_z += log_importance_weights.view(batch_size, batch_size, 1)

        log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
        log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)

        mi_loss = (log_q_zx - log_q_z).mean()
        tc_loss = (log_q_z - log_prod_q_z).mean()
        kld_loss = (log_prod_q_z - log_p_z).mean()

        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.training:
            self.num_iter += 1
            anneal_rate = min(0 + 1 * self.num_iter / self.anneal_steps, 1)
        else:
            anneal_rate = 1.

        loss = recons_loss / batch_size + \
               self.alpha * mi_loss + \
               weight * (self.beta * tc_loss +
                         anneal_rate * self.gamma * kld_loss)

        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'KLD': kld_loss,
                'TC_Loss': tc_loss,
                'MI_Loss': mi_loss}
