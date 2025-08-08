
from torchvision import transforms
from torchvision.transforms import InterpolationMode

_DEFAULT_MU = .5
_DEFAULT_SIGMA = .5


class Augmentation:
    train_transform = None
    val_transform = None

    @classmethod
    def _get_train_transform(cls, augment, res):
        if augment:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(res, scale=(0.5, 2), ratio=(1, 1),
                    interpolation=InterpolationMode.BILINEAR),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.ToTensor(),
                transforms.Normalize(_DEFAULT_MU, _DEFAULT_SIGMA),
            ])
        else:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(res, pad_if_needed=True),
                transforms.RandomResizedCrop(res, scale=(1, 1), ratio=(1, 1), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(_DEFAULT_MU, _DEFAULT_SIGMA),
            ])

    @classmethod
    def calc_transform(cls, augment, res):
        cls.train_transform = cls._get_train_transform(augment, res)
        cls.val_transform = cls._get_train_transform(False, res)
