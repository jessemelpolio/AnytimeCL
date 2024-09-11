import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import copy
import json

def dataset_prompt_template(dataset):
    templates = {
        "MNIST": ['a photo of the number: "{}".'],
        "SVHN": ['a photo of the number: "{}".'],
        "CIFAR10": ["a photo of a {}."],
        "CIFAR100": ["a photo of a {}."],
        "STL10": ["a photo of a {}."],
        "ImageNet": ["a photo of a {}."],
        "SUN397": ["a photo of a {}."],
        "Flowers102": ["a photo of a {}, a type of flower."],
        "FGVCAircraft": ["a photo of a {}, a type of aircraft."],
        "Food101": ["a photo of {}, a type of food."],
        "StanfordCars": ["a photo of a {}, a type of car."],
        "OxfordIIITPet": ["a photo of a {}, a type of pet."],
        "DTD": ["a photo of a {} texture."],
        "Places365LT": ["a photo of the {}, a type of place."],
        "iNaturalist": ["a photo of a {}, a type of species."],
        "PCAM": ["this is a photo of {}"],
        "EuroSAT": ["a centered satellite photo of {}."],
        "Resisc45": ["satellite photo of {}."],
        "UCF101": ["a video of a person doing {}."],
        "CLEVRCount": ["a photo of {} objects."],
        "CLEVRDist": ["a photo where the closest object is {} pixels away."]
    }
    if dataset not in templates:
        raise ValueError("Dataset not supported")
    return templates[dataset]

def standardize_dataset(dataset):
    dataset.prompt_template = dataset_prompt_template(dataset.__class__.__name__)
    print(f"Standardizing dataset for {dataset.__class__.__name__}...")

    class_files = {
        "Flowers102": "./data/dataset_classes/flowers102_classes.json",
        "UCF101": "./data/dataset_classes/ucf101_classes.json"
    }

    if dataset.__class__.__name__ in class_files:
        classes = json.load(open(class_files[dataset.__class__.__name__], "r"))
        dataset.classes = classes
        dataset.class_to_idx = {classes[i]: i for i in range(len(classes))}
    elif dataset.__class__.__name__ == "PCAM":
        dataset.classes = [
            "lymph node",
            "lymph node containing metastatic tumor tissue",
        ]
        dataset.class_to_idx = {dataset.classes[i]: i for i in range(len(dataset.classes))}

    assert hasattr(dataset, "classes"), f"Dataset {dataset.__class__.__name__} does not have classes"
    assert hasattr(dataset, "class_to_idx"), f"Dataset {dataset.__class__.__name__} does not have class_to_idx"

    if hasattr(dataset, "targets"):
        return dataset
    elif hasattr(dataset, "_labels"):
        dataset.targets = dataset._labels
    elif hasattr(dataset, "_samples"):
        dataset.targets = [sample[-1] for sample in dataset._samples]
    else:
        dataset.targets = [label for _, label in dataset]
    return copy.deepcopy(dataset)

def get_augment_transforms(inp_sz):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_augment = [torchvision.transforms.RandomResizedCrop(inp_sz), torchvision.transforms.RandomHorizontalFlip()]
    test_augment = [torchvision.transforms.Resize(inp_sz + 32), torchvision.transforms.CenterCrop(inp_sz)]

    train_augment_transform = Compose(train_augment + [ToTensor(), Normalize(mean, std)])
    test_augment_transform = Compose(test_augment + [ToTensor(), Normalize(mean, std)])

    return train_augment_transform, test_augment_transform

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_clip_transform(input_size):
    try:
        from torchvision.transforms import InterpolationMode
        BICUBIC = InterpolationMode.BICUBIC
    except ImportError:
        BICUBIC = Image.BICUBIC

    return Compose([
        Resize(input_size, interpolation=BICUBIC),
        CenterCrop(input_size),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def deal_with_dataset(args):
    dataset_list = args.datasets.split(",")
    return dataset_list, len(dataset_list)

def dataset_func(dataset):
    dataset_map = {
        "CIFAR10": torchvision.datasets.CIFAR10,
        "CIFAR100": torchvision.datasets.CIFAR100,
        "SVHN": torchvision.datasets.SVHN,
        "MNIST": torchvision.datasets.MNIST,
        "KMNIST": torchvision.datasets.KMNIST,
        "FashionMNIST": torchvision.datasets.FashionMNIST,
        "STL10": torchvision.datasets.STL10,
        "Caltech256": torchvision.datasets.Caltech256,
        "Omniglot": torchvision.datasets.Omniglot,
        "Flowers102": torchvision.datasets.Flowers102,
        "FGVCAircraft": torchvision.datasets.FGVCAircraft,
        "Food101": torchvision.datasets.Food101,
        "StanfordCars": torchvision.datasets.StanfordCars,
        "OxfordIIITPet": torchvision.datasets.OxfordIIITPet,
        "DTD": torchvision.datasets.DTD,
        "PCAM": torchvision.datasets.PCAM,
        "iNaturalist": "data.dataset_classes.inaturalist.iNaturalist",
        "Places365LT": "data.dataset_classes.places365.Places365LT",
        "SUN397": "data.dataset_classes.sun397.SUN397",
        "EuroSAT": "data.dataset_classes.eurosat.EuroSAT",
        "ImageNet": "data.dataset_classes.imagenet.ImageNet",
        "UCF101": "data.dataset_classes.ucf101.UCF101",
        "Resisc45": "data.dataset_classes.resisc45.Resisc45",
        "CLEVRCount": "data.dataset_classes.clevr.CLEVRCount",
        "CLEVRDist": "data.dataset_classes.clevr.CLEVRDist",
    }
    if dataset not in dataset_map:
        raise ValueError("Dataset not supported")
    if isinstance(dataset_map[dataset], str):
        module, cls = dataset_map[dataset].rsplit(".", 1)
        return getattr(__import__(module, fromlist=[cls]), cls)
    return dataset_map[dataset]

def single_dataset_build(args, dataset, transform=None):
    if transform is None:
        if args.network_arc in ["clip", "rac"]:
            train_transform = test_transform = get_clip_transform(args.input_size)
        else:
            train_transform = test_transform = None
    else:
        train_transform = test_transform = transform

    df = dataset_func(dataset)
    if dataset in ["CIFAR10", "CIFAR100"]:
        train_set = df(root=args.data_root, train=True, download=True, transform=train_transform)
        test_set = df(root=args.data_root, train=False, download=True, transform=test_transform)
    elif dataset in ["Food101", "StanfordCars", "SVHN"]:
        train_set = df(root=args.data_root, split="train", download=True, transform=train_transform)
        test_set = df(root=args.data_root, split="test", download=True, transform=test_transform)
    elif dataset == "Flowers102":
        train_set = df(root=args.data_root, split="train", download=True, transform=train_transform)
        test_set = df(root=args.data_root, split="val", download=True, transform=test_transform)
    elif dataset == "OxfordIIITPet":
        train_set = df(root=args.data_root, split="trainval", download=True, transform=train_transform)
        test_set = df(root=args.data_root, split="test", download=True, transform=test_transform)
    elif dataset in ["Places365LT", "iNaturalist", "ImageNet", "CLEVRCount", "CLEVRDist", "Resisc45", "UCF101"]:
        train_set = df(root=args.data_root, split="train", transform=train_transform)
        test_set = df(root=args.data_root, split="val", transform=test_transform)
    else:
        train_set = df(root=args.data_root, split="train", download=True, transform=train_transform)
        test_set = df(root=args.data_root, split="val", download=True, transform=test_transform)

    return standardize_dataset(train_set), standardize_dataset(test_set)

