import argparse


def get_config():

    parser = argparse.ArgumentParser(description='Disentangled Generative Causal Representation (DEAR)')

    # Data settings
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='tree', choices=['celeba', 'pendulum', 'human', 'tree'])
    parser.add_argument('--data_dir', type=str, default='./data/c3dtree/', help='data directory')

    # Training settings
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr_g', type=float, default=5e-5)
    parser.add_argument('--lr_e', type=float, default=5e-5)
    parser.add_argument('--lr_d', type=float, default=1e-4)
    parser.add_argument('--lr_p', type=float, default=5e-5, help='lr of SCM prior network')
    parser.add_argument('--lr_a', type=float, default=1e-3, help='lr of adjacency matrix')
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--d_steps_per_iter', type=int, default=1, help='how many D updates per iteration')
    parser.add_argument('--g_steps_per_iter', type=int, default=1, help='how many G updates per iteration')
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=10)

    # Model settings
    parser.add_argument('--latent_dim', type=int, default=9)
    parser.add_argument('--sup_coef', type=float, default=1, help='coefficient of the supervised regularizer')
    parser.add_argument('--sup_prop', type=float, default=1, help='proportion of supervised labels')
    parser.add_argument('--sup_type', type=str, default='ce', choices=['ce', 'l2'])
    parser.add_argument('--labels', type=str, default='tre', help='name of the underlying structure')

    # Prior settings
    parser.add_argument('--prior', type=str, default='nlrscm', choices=['gaussian', 'uniform', 'linscm', 'nlrscm'],
                        help='latent prior p_z')

    # Encoder settings
    parser.add_argument('--enc_arch', type=str, default='resnet', choices=['resnet', 'resnet18', 'dcgan'],
                        help='encoder architecture')
    parser.add_argument('--enc_dist', type=str, default='gaussian', choices=['deterministic', 'gaussian', 'implicit'],
                        help='encoder distribution')
    parser.add_argument('--enc_fc_size', type=int, default=1024, help='number of nodes in fc layer of resnet')
    parser.add_argument('--enc_noise_dim', type=int, default=128)
    # Generator settings
    parser.add_argument('--dec_arch', type=str, default='sagan', choices=['sagan', 'dcgan'],
                        help='decoder architecture')
    parser.add_argument('--dec_dist', type=str, default='implicit', choices=['deterministic', 'gaussian', 'implicit'],
                        help='generator distribution')
    parser.add_argument('--g_conv_dim', type=int, default=32, help='base number of channels in encoder and generator')

    # Discriminator settings
    parser.add_argument('--dis_fc_size', type=int, default=512, help='number of nodes in fc layer of joint discriminator')
    parser.add_argument('--d_conv_dim', type=int, default=32, help='base number of channels in discriminator')


    # Pretrained model
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--ckpt_dir', type=str, default='')

    # Output and save
    parser.add_argument('--print_every', type=int, default=500)
    parser.add_argument('--sample_every', type=int, default=1)
    parser.add_argument('--sample_every_epoch', type=int, default=1)
    parser.add_argument('--save_model_every', type=int, default=5)
    parser.add_argument('--save_name', type=str, default='')
    parser.add_argument('--save_n_samples', type=int, default=64)
    parser.add_argument('--save_n_recons', type=int, default=32)
    parser.add_argument('--nrow', type=int, default=8)


    # loss
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--beta', type=float, default=4.)
    parser.add_argument('--gamma', type=float, default=1.)
    parser.add_argument('--reconstruction_loss', type=str, default='mse', choices=['mse', 'bce'])
    parser.add_argument('--use_mss', type=bool, default=True)


    args = parser.parse_args()

    return args
