import argparse
from cleanfid import fid
from core.base_dataset import BaseDataset
from models.metric import inception_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str,
                        default=r"D:\Datasets\CelebA-HQ\celeba-hq\images_val_png",
                        help='Ground truth images directory')
    parser.add_argument('-d', '--dst', type=str,
                        default=r"E:\Pycharm\Palette_Windows\experiments\test_1000_400_s30\results\test\1",
                        help='Generate images directory')
    ''' parser configs '''
    args = parser.parse_args()

    # fid_score = fid.compute_fid(args.src, args.dst, num_workers=0)  # batch_size=32
    is_mean, is_std = inception_score(BaseDataset(args.dst), cuda=True, batch_size=8, resize=True, splits=10)

    # print('FID: {}'.format(fid_score))
    print('IS:{} {}'.format(is_mean, is_std))
