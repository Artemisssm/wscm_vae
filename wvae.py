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

    def __init__(self, latent_dim=64, conv_dim=32, image_size=64,
                 enc_dist='gaussian', enc_arch='resnet', enc_fc_size=2048, enc_noise_dim=128, dec_dist='implicit',
                 prior='gaussian', num_label=None, A=None, reconstruction_loss='mse', use_mss=True, alpha=1, beta=4, gamma=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.enc_dist = enc_dist
        self.dec_dist = dec_dist
        self.prior_dist = prior
        self.num_label = num_label
        self.reconstruction_loss = reconstruction_loss
        self.use_mss = use_mss
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
        sample = torch.zeros((n * (dim+1), 3, self.image_size, self.image_size))  # 生成一个全零张量，并赋值给sample
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
        sample[n * (dim+1) - n:n * (dim+1), :, :, :] = self.decoder(z)
        return sample  # 返回sample

    def forward(self, x=None, z=None, recon=False, gen=False, infer_mean=True):
        # recon_mean is used for gaussian decoder which we do not use here.
        # Training Mode
        if x is not None and z is None:  # 如果x和z都不是None
            if self.enc_dist == 'gaussian':  # 如果self.enc_dist是'gaussian'
                z_mu, z_logvar = self.encoder(x)  # 调用self.encoder，得到隐变量的均值和对数方差，并赋值给z_mu和z_logvar
                z_logvar = torch.ones(z_logvar.shape, device=x.device)
            else:  # deterministic or implicit
                z_fake = self.encoder(x)  # 调用self.encoder，得到隐变量，并赋值给z_fake


            if 'scm' in self.prior_dist:  # 如果self.prior_dist中包含'scm'
                # in prior
                label_z_q = self.prior(z_mu[:, :self.num_label])  # 调用self.prior，得到z的前self.num_label个维度经过因果层后的输出，并赋值给label_z
                other_z_q = z_mu[:, self.num_label:]  # 得到z的剩余维度，并赋值给other_z
                z_mu_fake = torch.cat([label_z_q, other_z_q], dim=1)  # 把label_z和other_z在第二个维度上拼接起来，并赋值给z


            z_logvar_fake = torch.ones(z_logvar.shape, device=x.device)
            z = reparameterize(z_mu_fake, (z_logvar_fake / 2).exp())

            x_fake = self.decoder(z)  # 调用self.decoder，得到生成的图像，并赋值给x_fake

            if recon is True:
                return x_fake

            return x_fake, z, z_mu_fake, z_logvar_fake, z_mu, z_logvar, label_z_q

            # if recon == True:
            #     return x_fake

            # if 'scm' in self.prior_dist:  # 如果self.prior_dist中包含'scm'
            #     if self.enc_dist == 'gaussian' and infer_mean:  # 如果self.enc_dist是'gaussian'并且infer_mean为真
            #         return z_fake, x_fake, z, z_mu, z_logvar  # 返回z_fake, x_fake, z, z_mu
            #     else:  # 否则
            #         return z_fake, x_fake, z, None, z_logvar  # 返回z_fake, x_fake, z, None
            # return z_fake, x_fake, z_mu, z_logvar  # 返回z_fake, x_fake, z_mu



        # Generation Mode
        elif x is None and z is not None:  # 如果x是None而z不是None
            if 'scm' in self.prior_dist:  # 如果self.prior_dist中包含'scm'
                label_z = self.prior(z[:, :self.num_label])  # 调用self.prior，得到z的前self.num_label个维度经过因果层后的输出，并赋值给label_z
                other_z = z[:, self.num_label:]  # 得到z的剩余维度，并赋值给other_z
                z = torch.cat([label_z, other_z], dim=1)  # 把label_z和other_z在第二个维度上拼接起来，并赋值给z

            x_fake = self.decoder(z)
            return  x_fake

            # if gen == True:
            #     return x_fake

            # if self.enc_dist == 'gaussian':  # 如果self.enc_dist是'gaussian'
            #     z_mu, z_logvar = self.encoder(x_fake)  # 调用self.encoder，得到隐变量的均值和对数方差，并赋值给z_mu和z_logvar
            #     z_fake = reparameterize(z_mu, torch.exp(0.5 * z_logvar))
            # else:  # deterministic or implicit
            #     z_fake = self.encoder(x_fake)  # 调用self.encoder，得到隐变量，并赋值给z_fake
            #
            # return z_fake, z_mu, z_logvar, x_fake, z  # 返回调用self.decoder后的结果



    def _compute_log_gauss_density(self, z, mu, log_var):
        """element-wise computation"""
        return -0.5 * (
                torch.log(torch.tensor([2 * np.pi]).to(z.device))
                + log_var
                + (z - mu) ** 2 * torch.exp(-log_var)
        )
    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        """Compute importance weigth matrix for MSS
        Code from (https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py)
        """

        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[:: M + 1] = 1 / N
        W.view(-1)[1 :: M + 1] = strat_weight
        W[M - 1, 0] = strat_weight
        return W.log()

    def loss_function(self, recon_x, x, mu, log_var, z, dataset_size, discriminator):

        if self.reconstruction_loss == "mse":

            recon_loss = (
                    0.5
                    * F.mse_loss(
                discriminator(recon_x)[1],
                discriminator(x)[1],
                reduction="none",
            ).sum(dim=-1)
            )

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        log_q_z_given_x = self._compute_log_gauss_density(z, mu, log_var).sum(
            dim=-1
        )  # [B]

        log_prior = self._compute_log_gauss_density(
            z, torch.zeros_like(z), torch.zeros_like(z)
        ).sum(
            dim=-1
        )  # [B]

        log_q_batch_perm = self._compute_log_gauss_density(
            z.reshape(z.shape[0], 1, -1),
            mu.reshape(1, z.shape[0], -1),
            log_var.reshape(1, z.shape[0], -1),
        )  # [B x B x Latent_dim]

        if self.use_mss:
            logiw_mat = self._log_importance_weight_matrix(z.shape[0], dataset_size).to(
                z.device
            )
            log_q_z = torch.logsumexp(
                logiw_mat + log_q_batch_perm.sum(dim=-1), dim=-1
            )  # MMS [B]
            log_prod_q_z = (
                torch.logsumexp(
                    logiw_mat.reshape(z.shape[0], z.shape[0], -1) + log_q_batch_perm,
                    dim=1,
                )
            ).sum(
                dim=-1
            )  # MMS [B]

        else:
            log_q_z = torch.logsumexp(log_q_batch_perm.sum(dim=-1), dim=-1) - torch.log(
                torch.tensor([z.shape[0] * dataset_size]).to(z.device)
            )  # MWS [B]
            log_prod_q_z = (
                    torch.logsumexp(log_q_batch_perm, dim=1)
                    - torch.log(torch.tensor([z.shape[0] * dataset_size]).to(z.device))
            ).sum(
                dim=-1
            )  # MWS [B]

        mutual_info_loss = log_q_z_given_x - log_q_z
        TC_loss = log_q_z - log_prod_q_z
        dimension_wise_KL = log_prod_q_z - log_prior

        return recon_loss.mean()/dataset_size, (self.beta * TC_loss).mean()

        # return (
        #     (
        #             recon_loss
        #             + self.alpha * mutual_info_loss
        #             + self.beta * TC_loss
        #             + self.gamma * dimension_wise_KL
        #     ).mean(dim=0),
        #     recon_loss.mean(dim=0),
        #     (
        #             self.alpha * mutual_info_loss
        #             + self.beta * TC_loss
        #             + self.gamma * dimension_wise_KL
        #     ).mean(dim=0),
        # )


class BigJointDiscriminator(nn.Module):
    r'''Big joint discriminator based on SAGAN

    Args:
        latent_dim: latent dimension
        conv_dim: base number of channels
        image_size: image resolution
        fc_size: number of nodes in each fc layers
    '''
    def __init__(self, latent_dim=64, conv_dim=32, image_size=64, fc_size=1024):
        super().__init__()
        self.discriminator = Discriminator(conv_dim, image_size, in_channels=3, out_feature=True) # 创建一个判别器，并把它赋值给self.discriminator
        self.discriminator_z = Discriminator_MLP(latent_dim, fc_size) # 创建一个多层感知器的判别器，并把它赋值给self.discriminator_z
        self.discriminator_j = Discriminator_MLP(conv_dim * 16 + fc_size, fc_size) # 创建一个多层感知器的判别器，并把它赋值给self.discriminator_j

    def forward(self, x=None, z=None):
        if x is not None and z is not None:
            sx, feature_x = self.discriminator(x)
            sz, feature_z = self.discriminator_z(z)
            sxz, _ = self.discriminator_j(torch.cat((feature_x, feature_z), dim=1))
            return (sx + sz + sxz) / 3
        elif x is not None and z is None:
            sx, feature_x = self.discriminator(x)
            # sx_f, _ = self.discriminator_j(feature_x)
            return sx, feature_x
        elif x is None and z is not None:
            sz, feature_z = self.discriminator_z(z)
            # sx_f, _ = self.discriminator_j(feature_x)
            return sz, feature_z

