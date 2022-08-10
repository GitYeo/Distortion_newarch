import os
import csv
import numpy as np
import torch
from absl import flags, app
from torchvision import datasets, transforms
from model.discriminator import Discriminator
from model.encoder_decoder import EncoderDecoder
from model.necst import NECST
from model.generator import Generator
import utils
from losses import loss_map
from noise_layers.noiser import Noiser
from noise_layers.identity import Identity
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.dropout import Dropout
from noise_layers.cropout import Cropout
from noise_layers.crop import Crop
from noise_layers.resize import Resize
from noise_layers.hue import Hue
from noise_layers.gaussian_noise import Gaussian_Noise
from noise_layers.sat import Sat
from noise_layers.blur import Blur

###########################
FLAGS = flags.FLAGS
flags.DEFINE_string('name', 'exp', 'experiments name')
flags.DEFINE_string('resume', None, 'resume from checkpoint')
flags.DEFINE_string('dataset', '/home/yeochengyu/Documents/flickr', 'Dataset used')
flags.DEFINE_string('out_dir', "output/", "Folder output sample_image")
flags.DEFINE_integer('image_size', 128, "size of the images generated")
flags.DEFINE_integer('batch_size', 32, "size of batch")
flags.DEFINE_integer('nc', 3, "Channel Dimension of image")
flags.DEFINE_integer('seed', 1, "random seed")
flags.DEFINE_bool('cuda', True, 'Flag using GPU')
flags.DEFINE_integer('eval_every', 500, "validation bit error with this step")
flags.DEFINE_integer('print_every', 50, "print training information with this step")
flags.DEFINE_integer('save_every', 2000, "saving checkpoint with this step")
flags.DEFINE_integer('message_length', 30, "length of message")
flags.DEFINE_integer('redundant_length', 30, "length of message")
flags.DEFINE_integer('iter', 1000000, "number of iterations model trained")

### Decoder ###
flags.DEFINE_integer('decoder_channels', 64, "number of channels of decoder")
flags.DEFINE_integer('decoder_blocks', 7, "number of blocks of decoder")
### Encoder ###
flags.DEFINE_integer('encoder_channels', 64, "number of channels of encoder")
flags.DEFINE_integer('encoder_blocks', 4, "number of blocks of encoder")
### Discriminator ###
flags.DEFINE_integer('discriminator_channels', 64, "number of channels of discriminator")
flags.DEFINE_integer('discriminator_blocks', 3, "number of blocks of discriminator")
flags.DEFINE_enum('loss', 'ns', loss_map.keys(), "loss function")


def train():
    device = torch.device('cuda:0' if FLAGS.cuda and torch.cuda.is_available() else 'cpu')
    print("device_name :", torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print("using CPU")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((FLAGS.image_size, FLAGS.image_size), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop((FLAGS.image_size, FLAGS.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_images = datasets.ImageFolder(FLAGS.dataset + "/train", data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=FLAGS.batch_size, shuffle=True,
                                               num_workers=4)

    validation_images = datasets.ImageFolder(FLAGS.dataset + "/val", data_transforms['test'])
    validation_loader = torch.utils.data.DataLoader(validation_images, batch_size=FLAGS.batch_size,
                                                    shuffle=False, num_workers=4)

    looper = utils.infiniteloop(train_loader, FLAGS.message_length, device)

    ### Model ###
    net_Encdec = EncoderDecoder(FLAGS).to(device)
    net_Dis = Discriminator(FLAGS).to(device)
    noiser = Noiser(device)

    ### Optimizer ###
    optim_EncDec = torch.optim.Adam(net_Encdec.parameters())
    optim_Dis = torch.optim.Adam(net_Dis.parameters())

    ## Loading checkpoint ##
    if FLAGS.resume:
        checkpoint = torch.load(FLAGS.resume)
        net_Encdec.load_state_dict(checkpoint['enc-dec-model'])
        optim_EncDec.load_state_dict(checkpoint['enc-dec-optim'])
        net_Dis.load_state_dict(checkpoint['discrim-model'])
        optim_Dis.load_state_dict(checkpoint['discrim-optim'])
        start = int(checkpoint['iter']) + 1
        FLAGS.out_dir = FLAGS.resume.split("checkpoint")[0]
        print("Loaded checkpoint from iter:{} in dir{}".format(start, FLAGS.out_dir))

    else:
        start = 1

    ## Define loss functions ##
    dis_loss = loss_map[FLAGS.loss]
    mse_loss = torch.nn.MSELoss()

    for i in range(start, FLAGS.iter):
        images, messages = next(looper)
        ########### training discriminator ###############
        optim_Dis.zero_grad()
        pred_real = net_Dis(images)
        encoded_images, decoded_messages = net_Encdec(images, messages, None, None, identity=True, noiser=noiser)
        pred_fake = net_Dis(encoded_images.detach())
        loss_fake, loss_real = dis_loss(pred_fake, pred_real)
        loss_D = (loss_real + loss_real) * 1.0
        loss_D.backward()
        optim_Dis.step()

        ########### training Encoder Decoder ###############
        optim_EncDec.zero_grad()
        pred_fake = net_Dis(encoded_images)
        loss_fake = dis_loss(pred_fake)
        enc_dec_image_loss = mse_loss(encoded_images, images)
        enc_dec_message_encloss = mse_loss(decoded_messages, messages)
        # loss_ED = 0.01 * loss_fake + 6.0 * enc_dec_image_loss + 1.0 * enc_dec_message_encloss
        loss_ED = 0.001 * loss_fake + 0.7 * enc_dec_image_loss + 1.0 * enc_dec_message_encloss
        loss_ED.backward()
        optim_EncDec.step()

        if i == 1 or i % FLAGS.print_every == 0:
            net_Encdec.eval()
            net_Dis.eval()
            with torch.no_grad():
                decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
                bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                            messages.shape[0] * messages.shape[1])
                print("\n########## Iteration:{} ##########\n".format(i))
                print("loss_Discim            :", loss_D.item())
                print("loss_Encdec            :", loss_ED.item())
                print("encdec_message_encloss :", enc_dec_message_encloss.item())
                print("encdec_image_loss      :", enc_dec_image_loss.item())
                print("bitwise_avg_err        :", bitwise_avg_err)
                print("\n###################################\n")

            net_Encdec.train()
            net_Dis.train()

        if i == 1 or i % FLAGS.eval_every == 0:
            losses_accu = {
                'Identity_err': 0.00,
                'Crop_err': 0.00,
                'Cropout_err': 0.00,
                'Dropout_err': 0.00,
                'Jpeg_err': 0.00,
                'Resize_err': 0.00,
                'Gaussian_err': 0.00,
                'Blur_err': 0.00,
                'Sat_err': 0.00,
                'Hue_err': 0.00}
            count = 0
            net_Encdec.eval()
            net_Dis.eval()
            with torch.no_grad():
                for images, _ in validation_loader:
                    count += 1
                    images = images.to(device)
                    messages = torch.Tensor(np.random.choice([0, 1], (images.shape[0], FLAGS.message_length))).to(
                        device)
                    for noise in noiser.noise_layers:
                        encoded_images = net_Encdec.encoder(images, messages)
                        noised_and_cover = noise([encoded_images, images])
                        noised_images = noised_and_cover[0]
                        decoded_messages = net_Encdec.decoder(noised_images)
                        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
                        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                                    messages.shape[0] * messages.shape[1])

                        if isinstance(noise, Identity):
                            losses_accu['Identity_err'] += bitwise_avg_err

                        elif isinstance(noise, Crop):
                            losses_accu['Crop_err'] += bitwise_avg_err

                        elif isinstance(noise, Cropout):
                            losses_accu['Cropout_err'] += bitwise_avg_err

                        elif isinstance(noise, Dropout):
                            losses_accu['Dropout_err'] += bitwise_avg_err

                        elif isinstance(noise, JpegCompression):
                            losses_accu['Jpeg_err'] += bitwise_avg_err

                        elif isinstance(noise, Resize):
                            losses_accu['Resize_err'] += bitwise_avg_err

                        elif isinstance(noise, Hue):
                            losses_accu['Hue_err'] += bitwise_avg_err

                        elif isinstance(noise, Blur):
                            losses_accu['Blur_err'] += bitwise_avg_err

                        elif isinstance(noise, Gaussian_Noise):
                            losses_accu['Gaussian_err'] += bitwise_avg_err

                        elif isinstance(noise, Sat):
                            losses_accu['Sat_err'] += bitwise_avg_err

            losses_accu['Identity_err'] /= count
            losses_accu['Crop_err'] /= count
            losses_accu['Cropout_err'] /= count
            losses_accu['Dropout_err'] /= count
            losses_accu['Jpeg_err'] /= count
            losses_accu['Resize_err'] /= count
            losses_accu['Hue_err'] /= count
            losses_accu['Blur_err'] /= count
            losses_accu['Gaussian_err'] /= count
            losses_accu['Sat_err'] /= count

            encoded_images = net_Encdec.encoder(images, messages)

            utils.save_images(images.cpu()[:8, :, :, :],
                              encoded_images[:8, :, :, :].cpu(),
                              i,
                              os.path.join(FLAGS.out_dir, 'images'), resize_to=(128, 128), imgtype="enc")

            print("\n########## Iteration:{} ##########\n".format(i))
            print("Identity_err     :", round(losses_accu['Identity_err'], 6))
            print("Crop_err         :", round(losses_accu['Crop_err'], 6))
            print("Cropout_err      :", round(losses_accu['Cropout_err'], 6))
            print("Dropout_err      :", round(losses_accu['Dropout_err'], 6))
            print("Jpeg_err         :", round(losses_accu['Jpeg_err'], 6))
            print("Resize_err       :", round(losses_accu['Resize_err'], 6))
            print("Gaussian_err     :", round(losses_accu['Gaussian_err'], 6))
            print("Blur_err         :", round(losses_accu['Blur_err'], 6))
            print("Hue_err          :", round(losses_accu['Hue_err'], 6))
            print("Sat_err          :", round(losses_accu['Sat_err'], 6))
            print("\n###################################\n")

            with open(FLAGS.out_dir + "/validation.csv", 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if i == 1:
                    row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()]
                    writer.writerow(row_to_write)
                row_to_write = [i] + ['{:.4f}'.format(loss_avg) for loss_avg in losses_accu.values()]
                writer.writerow(row_to_write)

            net_Encdec.train()
            net_Dis.train()

        if i % FLAGS.save_every == 0:
            ## saving checkpoint ##
            checkpoint = {
                'enc-dec-model': net_Encdec.state_dict(),
                'enc-dec-optim': optim_EncDec.state_dict(),
                'discrim-model': net_Dis.state_dict(),
                'discrim-optim': optim_Dis.state_dict(),
                'iter': i
            }
            torch.save(checkpoint, FLAGS.out_dir + "checkpoint/" + "hidden_iter{}.pyt".format(i))
            print('iter {} Saving checkpoint done.'.format(i))


def main(argv):
    # utils.set_seed(FLAGS.seed)

    if not os.path.exists(FLAGS.out_dir):
        os.mkdir(FLAGS.out_dir)
    run = 0
    while os.path.exists(FLAGS.out_dir + FLAGS.name + str(run) + "/"):
        run += 1
    FLAGS.out_dir = FLAGS.out_dir + FLAGS.name + str(run) + "/"
    os.mkdir(FLAGS.out_dir)
    os.mkdir(FLAGS.out_dir + "images")
    os.mkdir(FLAGS.out_dir + "checkpoint")
    train()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
