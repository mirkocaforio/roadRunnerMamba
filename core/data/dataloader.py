# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from collections import Counter
import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader

class INatDataset(ImageFolder):
    def __init__(
        self, 
        root,
        train=True,
        year=2018,
        transform=None,
        target_transform=None,
        category='name',
        loader=default_loader
    ):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, is_test, args):
    #transform = build_transform(is_train, args)
    
    """
    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(
            args.data_path,
            train=is_train,
            year=2018,
            category=args.inat_category,
            transform=transform
        )
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(
            args.data_path,
            train=is_train,
            year=2019,
            category=args.inat_category,
            transform=transform
        )
        nb_classes = dataset.nb_classes
    elif args.data_set == 'RSCD':
        root = os.path.join(args.data_path, 'train' if is_train else 'val' if not is_test else 'test')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = len(dataset.classes)
    else:
        raise NotImplementedError(f"Unknown dataset: {args.data_set}")

    return dataset, nb_classes
    """
    if args.data_set == 'RSCD':
        root = os.path.join(args.data_path, 'train' if is_train else 'val' if not is_test else 'test')
        #dataset = datasets.ImageFolder(root, transform=build_transform(is_train, args.data_path, args))
        dataset = datasets.ImageFolder(root, transform=build_transform(is_train, args))
        nb_classes = len(dataset.classes)
    else:
        raise NotImplementedError(f"Unknown dataset: {args.data_set}")
    
    return dataset, nb_classes

#ORIGINALE
def build_transform(is_train, args):
    resize_im = args.img_size > 32
    #mean, std = _load_dataset_stats(args.data_path)
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.img_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.img_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.img_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.img_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    #t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

"""
def build_transform(is_train: bool, data_path: str, args) -> transforms.Compose:
    #Transform ottimizzato per immagini stradali 240×360.
    
    #Strategia:
    #- Resize diretto a (224, 224) invece di resize+crop: mantiene tutto
    #  il contenuto dell'immagine, la deformazione geometrica è accettabile
    #  per classificazione di texture (dry/wet/ice non dipendono dalla forma).
    #- Training: augmentation conservative focalizzate su variazioni
    #  realistiche di illuminazione e condizioni atmosferiche.
    #- Val/Test: pipeline deterministica, nessuna augmentation.
    
    # Normalizzazione ImageNet — ragionevole se usi un backbone pretrained
    #mean = getattr(args, 'mean', IMAGENET_DEFAULT_MEAN)
    #std  = getattr(args, 'std',  IMAGENET_DEFAULT_STD)
    #mean, std = compute_mean_std_single_pass(data_path=data_path, img_size=args.img_size)
    mean, std = _load_dataset_stats(data_path)

    if is_train:
        transform = transforms.Compose([
            # Resize diretto: niente crop, mantieni tutto il contenuto
            transforms.Resize((args.img_size, args.img_size)),

            # Flip orizzontale: realistico per strade
            transforms.RandomHorizontalFlip(p=0.5),

            # Flip verticale: usalo SOLO se nel tuo dataset ci sono
            # immagini da entrambe le direzioni di marcia. Altrimenti
            # commentalo — prospettiva e orizzonte diventerebbero irrealistici.
            # transforms.RandomVerticalFlip(p=0.3),

            # Variazioni di luminosità/contrasto/saturazione: simulano
            # diverse condizioni di luce (sole, nuvoloso, alba/tramonto).
            # Questi sono i più utili per superfici stradali.
            transforms.ColorJitter(
                brightness=0.3,   # ±30% luminosità
                contrast=0.3,     # ±30% contrasto
                saturation=0.2,   # ±20% saturazione (ghiaccio/neve cambiano poco)
                hue=0.05,         # variazione hue minima: non vuoi verde→blu
            ),

            # Rotazioni piccole: utile se la camera non è sempre perfettamente
            # allineata. Valori grandi (>15°) rendono la scena irrealistica.
            transforms.RandomRotation(degrees=5),

            # Gaussian blur: simula pioggia, nebbia, sporco sull'obiettivo.
            # Molto utile per distinguere wet/water in condizioni avverse.
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))
            ], p=0.2),

            # Grayscale occasionale: forza il modello a non dipendere
            # solo dal colore per distinguere ice/snow/wet.
            transforms.RandomGrayscale(p=0.05),

            transforms.ToTensor(),
            transforms.Normalize(mean, std),

            # Random erasing: maschera patch casuali dell'immagine,
            # simula occlusioni (veicoli, riflessi). Valore basso
            # per non distruggere texture importante.
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])

    else:
        # Val e test: deterministico, nessuna augmentation
        transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    return transform
"""

def build_weighted_sampler(dataset) -> WeightedRandomSampler:
    targets = [s[1] for s in dataset.samples]  # ImageFolder ha .samples
    counts  = Counter(targets)
    weights = [1.0 / counts[t] for t in targets]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )
    
def compute_mean_std_single_pass(data_path: str, img_size: int = 224, batch_size: int = 64, num_workers: int = 4):
    """
    Versione a passata singola — più efficiente e numericamente stabile.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    #dataset = datasets.ImageFolder(root=f"{data_path}/train", transform=transform)
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Accumula somma e somma dei quadrati
    channel_sum    = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    n_pixels       = 0

    for imgs, _ in loader:
        B, C, H, W  = imgs.shape
        n_pixels    += B * H * W
        channel_sum    += imgs.sum(dim=[0, 2, 3])       # somma su B, H, W
        channel_sum_sq += (imgs ** 2).sum(dim=[0, 2, 3])

    mean = channel_sum / n_pixels
    std  = torch.sqrt(channel_sum_sq / n_pixels - mean ** 2)

    print(f"Dataset: {len(dataset)} immagini, {n_pixels:,} pixel per canale")
    print(f"Mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"Std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")

    return mean.tolist(), std.tolist()

def _load_dataset_stats(data_path: str, fallback_to_imagenet: bool = True):
    """
    Carica mean e std dal file pre-calcolato dataset_stats.json.
    Se non esiste, usa i valori ImageNet come fallback con un warning.
    NON ricalcola mai al volo — troppo lento su dataset grandi.
    """
    import json

    stats_file = os.path.join(data_path, 'dataset_stats.json')

    if os.path.exists(stats_file):
        with open(stats_file) as f:
            stats = json.load(f)
        print(f"✓ Dataset stats loaded from {stats_file}")
        print(f"  Mean: {stats['mean']}")
        print(f"  Std:  {stats['std']}")
        return stats['mean'], stats['std']

    if fallback_to_imagenet:
        print(
            f"⚠️  {stats_file} not found. "
            f"Using ImageNet defaults (mean/std may be suboptimal). "
            f"Run compute_mean_std_single_pass() once to generate it."
        )
        return list(IMAGENET_DEFAULT_MEAN), list(IMAGENET_DEFAULT_STD)

    raise FileNotFoundError(
        f"Dataset stats file not found: {stats_file}\n"
        f"Run compute_mean_std_single_pass(data_path='{data_path}') "
        f"to generate it."
    )