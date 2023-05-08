from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T
from torchtoolbox.transform import Cutout


plt.rcParams["savefig.bbox"] = "tight"
orig_img = Image.open(Path("data/utk_face/images") / "22_1_3_20170119163702901.jpg")
# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(0)


def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title="Original image")
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])
    plt.tight_layout()
    plt.savefig("example.png")


def single_augmetation_plot():
    img_size = 64
    load_size = 72
    crop = [
        T.Compose(
            [
                T.Resize(load_size),
                T.CenterCrop(img_size),
                T.RandomResizedCrop(size=img_size, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
            ]
        )(orig_img)
        for _ in range(4)
    ]
    cutout = [
        T.Compose([T.Resize(load_size), T.CenterCrop(img_size), Cutout()])(orig_img)
        for _ in range(4)
    ]
    rotate = [
        T.Compose(
            [
                T.Resize(load_size),
                T.CenterCrop(img_size),
                T.RandomRotation(degrees=(0, 180)),
            ]
        )(orig_img)
        for _ in range(4)
    ]
    color = [
        T.Compose(
            [
                T.Resize(load_size),
                T.CenterCrop(img_size),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
            ]
        )(orig_img)
        for _ in range(4)
    ]
    blur = [
        T.Compose(
            [
                T.Resize(load_size),
                T.CenterCrop(img_size),
                T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            ]
        )(orig_img)
        for _ in range(4)
    ]
    plot(
        [crop, cutout, rotate, color, blur],
        row_title=["crop", "cutout", "rotate", "color", "blur"],
    )


def combined_augmentation_plot():
    img_size = 64
    load_size = 72
    policies = [
        T.AutoAugmentPolicy.CIFAR10,
        T.AutoAugmentPolicy.IMAGENET,
        T.AutoAugmentPolicy.SVHN,
    ]
    auto = [T.AutoAugment(policy) for policy in policies]

    rand = [
        T.Compose([T.Resize(load_size), T.CenterCrop(img_size), T.RandAugment()])(
            orig_img
        )
        for _ in range(4)
    ]

    trivial = [
        T.Compose(
            [T.Resize(load_size), T.CenterCrop(img_size), T.TrivialAugmentWide()]
        )(orig_img)
        for _ in range(4)
    ]

    imgs = [
        [
            T.Compose([T.Resize(load_size), T.CenterCrop(img_size), augmenter])(
                orig_img
            )
            for _ in range(4)
        ]
        for augmenter in auto
    ]
    imgs.append(rand)
    imgs.append(trivial)
    row_title = [str(policy).split(".")[-1] for policy in policies]
    row_title.append("Rand")
    row_title.append("Trivial")
    plot(imgs, row_title=row_title)


if __name__ == "__main__":
    combined_augmentation_plot()
