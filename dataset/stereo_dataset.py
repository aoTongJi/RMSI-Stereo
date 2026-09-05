import os
import copy
import torch
import numpy as np
import imageio.v3 as imageio
from glob import glob
from util.reader import *
from util.augmentor import Augmentor

DATASET_ROOT = '/mnt/ssd512/datasets'


def _resolve_root(root):
    root = os.path.expanduser(root)
    if os.path.exists(root):
        return root

    local_root = os.path.join(DATASET_ROOT, os.path.basename(os.path.normpath(root)))
    if os.path.exists(local_root):
        return local_root

    raise FileNotFoundError(f'Dataset root does not exist: {root} (also tried {local_root})')


def _replace_path_component(path, old, new):
    return os.sep.join(new if part == old else part for part in path.split(os.sep))


def _read_rgb(filename):
    image = imageio.imread(filename).astype(np.float32)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    else:
        image = image[..., :3]

    return image


def _check_triplets(name, left_list, right_list, disp_list):
    if len(left_list) != len(right_list) or len(left_list) != len(disp_list):
        raise RuntimeError(
            f'{name} has mismatched files: '
            f'{len(left_list)} left, {len(right_list)} right, {len(disp_list)} disparity'
        )
    if len(left_list) == 0:
        raise RuntimeError(f'{name} found no samples. Please check the dataset root and split.')

    missing = []
    for filename in right_list + disp_list:
        if not os.path.exists(filename):
            missing.append(filename)
            if len(missing) >= 5:
                break

    if missing:
        raise FileNotFoundError(f'{name} has missing paired files, for example: {missing}')


def _add_triplets(dataset, left_list, right_list, disp_list):
    for left, right, disp in zip(left_list, right_list, disp_list):
        dataset.image_list += [[left, right]]
        dataset.disp_list += [disp]


class StereoDataset(torch.utils.data.Dataset):
    def __init__(self, sparse=False, aug_params=None, reader=None, mask=None):
        self.augmentor = None
        self.reader = reader
        self.mask = mask
        self.image_list = []
        self.disp_list = []

        if aug_params:
            self.augmentor = Augmentor(sparse, aug_params)

    def __len__(self):
        return len(self.image_list)

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disp_list = v * copy_of_self.disp_list

        return copy_of_self

    def __getitem__(self, index):
        left = _read_rgb(self.image_list[index][0])
        right = _read_rgb(self.image_list[index][1])
        disp, valid = self.reader(self.disp_list[index], self.mask)

        if self.augmentor:
            left, right, disp, valid = self.augmentor(name=self.image_list[index][0], left=left, right=right, disp=disp, valid=valid)

        left = torch.from_numpy(left).permute(2, 0, 1).float()
        right = torch.from_numpy(right).permute(2, 0, 1).float()
        disp = torch.from_numpy(disp)[None].float()
        valid = torch.from_numpy(valid)[None].float()

        return self.image_list[index][0], left, right, disp, valid

class SceneFlow(StereoDataset):
    def __init__(self, aug_params=None, root='/mnt/ssd512/datasets/sceneflow', dstype='frames_finalpass', things_test=False, mask=None):
        super(SceneFlow, self).__init__(sparse=False, aug_params=aug_params, reader=sceneflow_disp_reader, mask=mask)
        self.root = _resolve_root(root)
        self.dstype = dstype

        if things_test:
            self._add_things('TEST')
        else:
            self._add_things('TRAIN')
            self._add_monkaa()
            self._add_driving()

        if len(self.image_list) == 0:
            raise RuntimeError(f'SceneFlow found no samples under {self.root}')

    def _subset_root(self, subset):
        subset_root = os.path.join(self.root, subset)
        return subset_root if os.path.isdir(subset_root) else self.root

    def _add_sceneflow_files(self, name, left_list):
        right_list = [_replace_path_component(img, 'left', 'right') for img in left_list]
        disp_list = [
            os.path.splitext(_replace_path_component(img, self.dstype, 'disparity'))[0] + '.pfm'
            for img in left_list
        ]

        _check_triplets(name, left_list, right_list, disp_list)
        _add_triplets(self, left_list, right_list, disp_list)

    def _add_things(self, split='TRAIN'):
        root = self._subset_root('flying')
        left_list = sorted(glob(os.path.join(root, self.dstype, split, '*/*/left/*.png')))
        self._add_sceneflow_files(f'SceneFlow FlyingThings3D {split}', left_list)

    def _add_monkaa(self):
        root = self._subset_root('monkaa')
        left_list = sorted(glob(os.path.join(root, self.dstype, '*/left/*.png')))
        self._add_sceneflow_files('SceneFlow Monkaa', left_list)

    def _add_driving(self):
        root = self._subset_root('driving')
        left_list = sorted(glob(os.path.join(root, self.dstype, '*/*/*/left/*.png')))
        self._add_sceneflow_files('SceneFlow Driving', left_list)

class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root='/mnt/ssd512/datasets/kitti', year='2015', split='training', mask='all'):
        super(KITTI, self).__init__(sparse=True, aug_params=aug_params, reader=kitti_disp_reader, mask=mask)
        root = _resolve_root(root)
        left_list, right_list, disp_list = [], [], []

        if year == '2012':
            left_list = sorted(glob(os.path.join(root, year, split, 'colored_0/*_10.png')))
            right_list = sorted(glob(os.path.join(root, year, split, 'colored_1/*_10.png')))
            if split == 'training':
                disp_list = sorted(glob(os.path.join(root, year, 'training', 'disp_occ/*_10.png')))
            else:
                disp_list = [os.path.join(root, year, 'training', 'disp_occ/000000_10.png')] * len(left_list)

        elif year == '2015':
            left_list = sorted(glob(os.path.join(root, year, split, 'image_2/*_10.png')))
            right_list = sorted(glob(os.path.join(root, year, split, 'image_3/*_10.png')))
            if split == 'training':
                disp_list = sorted(glob(os.path.join(root, year, 'training', 'disp_occ_0/*_10.png')))
            else:
                disp_list = [os.path.join(root, year, 'training', 'disp_occ_0/000000_10.png')] * len(left_list)

        elif year == 'all':
            left_list = sorted(glob(os.path.join(root, '2012', split, 'colored_0/*_10.png')))
            right_list = sorted(glob(os.path.join(root, '2012', split, 'colored_1/*_10.png')))
            if split == 'training':
                disp_list = sorted(glob(os.path.join(root, '2012', 'training', 'disp_occ/*_10.png')))
            else:
                disp_list = [os.path.join(root, '2012', 'training', 'disp_occ/000000_10.png')] * len(left_list)

            left_list += sorted(glob(os.path.join(root, '2015', split, 'image_2/*_10.png')))
            right_list += sorted(glob(os.path.join(root, '2015', split, 'image_3/*_10.png')))
            if split == 'training':
                disp_list += sorted(glob(os.path.join(root, '2015', 'training', 'disp_occ_0/*_10.png')))
            else:
                n_2015 = len(sorted(glob(os.path.join(root, '2015', split, 'image_2/*_10.png'))))
                disp_list += [os.path.join(root, '2015', 'training', 'disp_occ_0/000000_10.png')] * n_2015
        else:
            raise ValueError(f'Unsupported KITTI year: {year}')

        _check_triplets(f'KITTI {year} {split}', left_list, right_list, disp_list)
        _add_triplets(self, left_list, right_list, disp_list)

class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='/mnt/ssd512/datasets/middlebury', year='MiddEval3', split='training', resolution='H', mask='noc'):
        super(Middlebury, self).__init__(sparse=True, aug_params=aug_params, reader=middlebury_disp_reader, mask=mask)
        root = _resolve_root(root)
        left_list, right_list, disp_list = [], [], []

        if year == 'MiddEval3':
            split_dir = split + resolution
            midd_dirs = [
                os.path.join(root, year, split_dir),
                os.path.join(root, f'MiddEval3-data-{resolution}', year, split_dir),
            ]
            midd_root = next((path for path in midd_dirs if glob(os.path.join(path, '*/im0.png'))), midd_dirs[0])

            left_list = sorted(glob(os.path.join(midd_root, '*/im0.png')))
            right_list = [os.path.join(os.path.dirname(left), 'im1.png') for left in left_list]
            disp_list = [os.path.join(os.path.dirname(left), 'disp0GT.pfm') for left in left_list]
        elif year == '2021':
            left_list = sorted(glob(os.path.join(root, year, 'data', '*/im0.png')))
            right_list = [os.path.join(os.path.dirname(left), 'im1.png') for left in left_list]
            disp_list = [os.path.join(os.path.dirname(left), 'disp0.pfm') for left in left_list]
        else:
            raise ValueError(f'Unsupported Middlebury year: {year}')

        _check_triplets(f'Middlebury {year} {split}', left_list, right_list, disp_list)
        _add_triplets(self, left_list, right_list, disp_list)

class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='/mnt/ssd512/datasets/eth3d', split='training', mask='noc'):
        super(ETH3D, self).__init__(sparse=True, aug_params=aug_params, reader=eth3d_disp_reader, mask=mask)
        root = _resolve_root(root)

        left_list = sorted(glob(os.path.join(root, f'two_view_{split}', '*/im0.png')))
        right_list = [os.path.join(os.path.dirname(left), 'im1.png') for left in left_list]
        disp_list = [
            os.path.join(root, 'two_view_training_gt', os.path.basename(os.path.dirname(left)), 'disp0GT.pfm')
            for left in left_list
        ]

        _check_triplets(f'ETH3D {split}', left_list, right_list, disp_list)
        _add_triplets(self, left_list, right_list, disp_list)

class DrivingStereo(StereoDataset):
    def __init__(self, aug_params=None, root='/data/datasets/drivingstereo', split='cloudy', mask=None, resolution='H'):
        super().__init__(sparse=True, aug_params=aug_params, reader=drivingstereo_disp_reader, mask=mask)
        assert os.path.exists(root)

        if resolution == 'F':
            res = 'full'
        elif resolution == 'H':
            res = 'half'

        left_list = sorted(glob(os.path.join(root, split, f'left-image-{res}-size/*.jpg')))
        right_list = sorted(glob(os.path.join(root, split, f'right-image-{res}-size/*.jpg')))
        disp_list = sorted(glob(os.path.join(root, split, f'disparity-map-{res}-size/*.png')))

        assert len(left_list) == len(right_list) == len(disp_list)

        for _, (left, right, disp) in enumerate(zip(left_list, right_list, disp_list)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]

class Booster(StereoDataset):
    def __init__(self, aug_params=None, root='/data/datasets/booster_q', split='train', light='balanced', mask=None):
        super().__init__(sparse=True, aug_params=aug_params, reader=booster_disp_reader, mask=mask)
        assert os.path.exists(root)

        left_list = sorted(glob(os.path.join(root, split, light, '*/camera_00/*.png')))
        right_list = sorted(glob(os.path.join(root, split, light, '*/camera_02/*.png')))

        assert len(left_list) == len(right_list)

        for _, (left, right) in enumerate(zip(left_list, right_list)):
            self.image_list += [[left, right]]
            self.disp_list += [os.path.join(os.path.dirname(left.replace('camera_00', '')), 'disp_00.npy')]

class FoundationStereo(StereoDataset):
    def __init__(self, aug_params=None, root='/data/datasets/foundationstereo', mask=None):
        super().__init__(sparse=False, aug_params=aug_params, reader=foundationstereo_disp_reader, mask=mask)
        assert os.path.exists(root)

        left_list = sorted(glob(os.path.join(root, '*/dataset/data/left/rgb/*.jpg')))
        right_list = sorted(glob(os.path.join(root, '*/dataset/data/right/rgb/*.jpg')))
        disp_list = sorted(glob(os.path.join(root, '*/dataset/data/left/disparity/*.png')))

        assert len(left_list) == len(right_list) == len(disp_list)

        for _, (left, right, disp) in enumerate(zip(left_list, right_list, disp_list)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]

class TartanAir(StereoDataset):
    def __init__(self, aug_params=None, root='/data/datasets/tartanair', mask=None):
        super().__init__(sparse=False, aug_params=aug_params, reader=tartanair_disp_reader, mask=mask)
        assert os.path.exists(root)

        left_list = sorted(glob(os.path.join(root, '*/*/*/*/image_left/*.png')))
        right_list = sorted(glob(os.path.join(root, '*/*/*/*/image_right/*.png')))
        disp_list = sorted(glob(os.path.join(root, '*/*/*/*/depth_left/*.npy')))

        assert len(left_list) == len(right_list) == len(disp_list)

        for _, (left, right, disp) in enumerate(zip(left_list, right_list, disp_list)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]

class CREStereo(StereoDataset):
    def __init__(self, aug_params=None, root='/data/datasets/crestereo', mask=None):
        super().__init__(sparse=False, aug_params=aug_params, reader=crestereo_disp_reader, mask=mask)
        assert os.path.exists(root)

        left_list = sorted(glob(os.path.join(root, '*/*_left.jpg')))
        right_list = sorted(glob(os.path.join(root, '*/*_right.jpg')))
        disp_list = sorted(glob(os.path.join(root, '*/*_left.disp.png')))

        assert len(left_list) == len(right_list) == len(disp_list)

        for _, (left, right, disp) in enumerate(zip(left_list, right_list, disp_list)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]

class FallingThings(StereoDataset):
    def __init__(self, aug_params=None, root='/data/datasets/fallingthings', mask=None):
        super().__init__(sparse=False, aug_params=aug_params, reader=fallingthings_disp_reader, mask=mask)
        assert os.path.exists(root)

        left_list = sorted(glob(os.path.join(root, '**/*.left.jpg'), recursive=True))
        right_list = sorted(glob(os.path.join(root, '**/*.right.jpg'), recursive=True))
        disp_list = sorted(glob(os.path.join(root, '**/*.left.depth.png'), recursive=True))

        assert len(left_list) == len(right_list) == len(disp_list)

        for _, (left, right, disp) in enumerate(zip(left_list, right_list, disp_list)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]
