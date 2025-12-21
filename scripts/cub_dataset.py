# ============================
# CUB Dataset Definition
# ============================

from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# Default Vision Transformer-style preprocessing (ImageNet)
def get_default_vit_transform() -> transforms.Compose:
    """
    Default preprocessing typically used for Vision Transformers
    trained on ImageNet:

      - Resize shorter side to 256
      - Center crop 224x224
      - Convert to tensor
      - Normalize with ImageNet mean/std
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class CUBDataset(Dataset):
    """
    CUB-200-2011 Dataset Wrapper.

    This class will:
      - Read metadata files (images.txt, image_class_labels.txt, train_test_split.txt)
      - Build a list of (image_path, label)
      - Load and transform images for PyTorch models
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Constructor (runs when you do: CUBDataset(...))

        In this step, we:
          - validate the 'split' argument
          - store basic attributes (root, split, transform)
          - build paths to the three important CUB metadata files
          - check that those files actually exist
          - read the metadata files and build the samples list
        """

        # Call parent Dataset constructor (good practice)
        super().__init__()

        # 1) Validate 'split' argument: must be exactly "train" or "test"
        assert split in ["train", "test"], "split must be 'train' or 'test'"

        # 2) Store attributes on 'self' so other methods can use them later
        # Convert the root string to a Path object for easier path operations
        self.root: Path = Path(root)
        self.split: str = split

        # If no transform is given, use a sensible ViT-style default
        if transform is None:
            self.transform: Callable = get_default_vit_transform()
        else:
            self.transform = transform

        # 3) Build Path objects for the three main metadata files
        #    These files live directly inside the CUB_200_2011 root folder.
        self.images_file: Path = self.root / "images.txt"
        self.labels_file: Path = self.root / "image_class_labels.txt"
        self.split_file: Path = self.root / "train_test_split.txt"

        # 4) Basic sanity checks: make sure the files actually exist.
        #    If not, fail early with a clear error instead of weird bugs later.
        if not self.images_file.is_file():
            raise FileNotFoundError(f"Missing images.txt at: {self.images_file}")

        if not self.labels_file.is_file():
            raise FileNotFoundError(
                f"Missing image_class_labels.txt at: {self.labels_file}"
            )

        if not self.split_file.is_file():
            raise FileNotFoundError(
                f"Missing train_test_split.txt at: {self.split_file}"
            )

        # 5) Read the metadata files and populate dictionaries
        self.id_to_relpath: Dict[int, str] = self._read_images_file(self.images_file)
        self.id_to_label: Dict[int, int] = self._read_labels_file(self.labels_file)
        self.id_to_istrain: Dict[int, int] = self._read_split_file(self.split_file)
        
        # Optional: number of distinct classes (should be 200 for CUB)
        self.num_classes: int = len(set(self.id_to_label.values()))




        # 6) Build the samples list: filter by split and create (image_path, label) tuples
        #    is_training = 1 means train split, 0 means test split
        target_istrain = 1 if split == "train" else 0

        self.samples: List[Tuple[Path, int]] = []
        for img_id in self.id_to_relpath.keys():
            # Check if this image belongs to the requested split
            if self.id_to_istrain.get(img_id) == target_istrain:
                # Build full path to image: root/images/relative_path
                rel_path = self.id_to_relpath[img_id]
                img_path = self.root / "images" / rel_path

                # Extra safety: check that the image file really exists
                if not img_path.is_file():
                    raise FileNotFoundError(
                        f"Image file missing for id {img_id}: {img_path}"
                    )

                # Class labels in CUB are 1-indexed, convert to 0-indexed for PyTorch
                label = self.id_to_label[img_id] - 1

                # Store (image_path, label) in the samples list
                self.samples.append((img_path, label))

    def _read_images_file(self, path: Path) -> Dict[int, str]:
        """
        Read images.txt and return a dictionary:

            { image_id (int) : relative_image_path (str) }

        Example line in images.txt:
            1 001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg

        After reading, we want:
            id_to_relpath[1] = "001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg"
        """

        id_to_relpath: Dict[int, str] = {}

        with path.open("r") as f:
            # Loop over each line in the file
            for line in f:
                # Remove whitespace at the start/end, including the newline "\n"
                line = line.strip()

                # If the line is empty, skip it
                if not line:
                    continue

                # Split the line into two parts:
                #   img_id_str = "1"
                #   rel_path  = "001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg"
                img_id_str, rel_path = line.split(maxsplit=1)

                # Convert the image id from string to int: "1" -> 1
                img_id: int = int(img_id_str)

                # Store mapping in the dictionary
                id_to_relpath[img_id] = rel_path

        return id_to_relpath

    def _read_labels_file(self, path: Path) -> Dict[int, int]:
        """
        Read image_class_labels.txt and return a dictionary:

            { image_id (int) : class_id (int) }

        Example lines in image_class_labels.txt:
            1 1
            2 1
            3 1
            4 2

        Meaning:
            image 1 -> class 1
            image 2 -> class 1
            image 3 -> class 1
            image 4 -> class 2
        """

        # Empty dictionary to fill: keys = image IDs, values = class IDs
        id_to_label: Dict[int, int] = {}

        # Open the labels file
        with path.open("r") as f:
            for line in f:
                # Remove whitespace and newline
                line = line.strip()
                if not line:
                    continue  # skip empty lines, if any

                # Split into two parts: "image_id" and "class_id"
                img_id_str, class_id_str = line.split(maxsplit=1)

                # Convert both to integers
                img_id: int = int(img_id_str)
                class_id: int = int(class_id_str)

                # Store in dictionary
                id_to_label[img_id] = class_id

        return id_to_label

    def _read_split_file(self, path: Path) -> Dict[int, int]:
        """
        Read train_test_split.txt and return a dictionary:

            { image_id (int) : is_training_image (int: 1 or 0) }

        Example lines in train_test_split.txt:
            1 1   -> image 1 is a training image
            2 1   -> image 2 is a training image
            3 0   -> image 3 is a test image
        """

        id_to_istrain: Dict[int, int] = {}

        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Split into: "image_id" and "is_training_image"
                img_id_str, istrain_str = line.split(maxsplit=1)

                img_id: int = int(img_id_str)
                istrain: int = int(istrain_str)  # 1 = train, 0 = test

                id_to_istrain[img_id] = istrain

        return id_to_istrain

    def __len__(self) -> int:
        """Number of samples in this split."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Load one sample (image and label) by index.

        Args:
            index: Index of the sample to retrieve

        Returns:
            Tuple of (image_tensor, label) where:
                - image_tensor: Transformed image as a torch.Tensor
                - label: Class label (0-indexed integer)
        """
        # Get the image path and label for this index
        img_path, label = self.samples[index]

        # Load the image using PIL
        image = Image.open(img_path).convert("RGB")

        # Apply transform (should include ToTensor and normalization)
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Fallback: just convert to tensor if no transform provided
            image = transforms.ToTensor()(image)

        return image, label
