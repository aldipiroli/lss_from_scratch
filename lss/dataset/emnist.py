import torch
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset
from torchvision.datasets import EMNIST

EMNIST_LETTERS_MAP = {
    0: "SPACE",
    1: "A",
    2: "B",
    3: "C",
    4: "D",
    5: "E",
    6: "F",
    7: "G",
    8: "H",
    9: "I",
    10: "J",
    11: "K",
    12: "L",
    13: "M",
    14: "N",
    15: "O",
    16: "P",
    17: "Q",
    18: "R",
    19: "S",
    20: "T",
    21: "U",
    22: "V",
    23: "W",
    24: "X",
    25: "Y",
    26: "Z",
}


class EMNISTDataset(Dataset):
    def __init__(self, cfg, mode, logger):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.logger = logger
        self.letter_mapping = EMNIST_LETTERS_MAP
        self.inv_letter_mapping = {v: k for k, v in EMNIST_LETTERS_MAP.items()}

        self.root_dir = cfg["DATA"]["root_dir"]
        self.train = mode == "train"
        self.dataset = EMNIST(
            root=self.root_dir,
            split="byclass",
            train=self.train,
            download=True,
        )
        self.indices = [
            idx for idx, (_, label) in enumerate(self.dataset) if 10 <= label <= 35
        ]
        self.transform = T.Compose(
            [
                T.Resize((32, 32)),
                T.ToTensor(),
                T.Lambda(lambda x: torch.transpose(x, 1, 2)),
                T.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        self.num_classes = cfg["MODEL"]["num_classes"] - 1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if torch.rand(1).item() < 1 / self.num_classes:
            # emtpy image
            img = torch.full((1, 32, 32), -1.0)
            label = 0
        else:
            img, label = self.dataset[self.indices[idx]]
            label = label - 9
            img = self.transform(img)
        return img, label
