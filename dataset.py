import os.path
import torch
import torch.utils.data as data
from PIL import Image
import random
from random import randrange
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def rotate(img, rotate_index):
    '''
    :return: 8 version of rotating image
    '''
    if rotate_index == 0:
        return img
    if rotate_index == 1:
        return img.rotate(90)
    if rotate_index == 2:
        return img.rotate(180)
    if rotate_index == 3:
        return img.rotate(270)
    if rotate_index == 4:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index == 5:
        return img.rotate(90).transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index == 6:
        return img.rotate(180).transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index == 7:
        return img.rotate(270).transpose(Image.FLIP_TOP_BOTTOM)


class TrainLabeled(data.Dataset):
    def __init__(self, dataroot, phase, finesize):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize

        self.dir_A = os.path.join(self.root, self.phase + '/trainA')
        self.dir_B = os.path.join(self.root, self.phase + '/trainB')
        self.dir_C = os.path.join(self.root, self.phase + '/trainC')

        # image path
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.C_paths = sorted(make_dataset(self.dir_C))

        # transform
        self.transform = ToTensor()  # [0,1]

    def __getitem__(self, index):
        # A, B is the image pair, hazy, gt respectively
        A = Image.open(self.A_paths[index]).convert("RGB")
        B = Image.open(self.B_paths[index]).convert("RGB")
        C = Image.open(self.C_paths[index]).convert("RGB")
        # resize
        resized_a = A.resize((280, 280), Image.LANCZOS)
        resized_b = B.resize((280, 280), Image.LANCZOS)
        resized_c = C.resize((280, 280), Image.LANCZOS)
        # crop the training image into fineSize
        w, h = resized_a.size
        x, y = randrange(w - self.fineSize + 1), randrange(h - self.fineSize + 1)
        cropped_a = resized_a.crop((x, y, x + self.fineSize, y + self.fineSize))
        cropped_b = resized_b.crop((x, y, x + self.fineSize, y + self.fineSize))
        cropped_c = resized_c.crop((x, y, x + self.fineSize, y + self.fineSize))
        # rotate
        rotate_index = randrange(0, 8)
        rotated_a = rotate(cropped_a, rotate_index)
        rotated_b = rotate(cropped_b, rotate_index)
        rotated_c = rotate(cropped_c, rotate_index)
        # transform to (0, 1)
        tensor_a = self.transform(rotated_a)
        tensor_b = self.transform(rotated_b)
        tensor_c = self.transform(rotated_c)

        return tensor_a, tensor_b, tensor_c

    def __len__(self):
        return len(self.A_paths)


class TrainUnlabeled(data.Dataset):
    def __init__(self, dataroot, phase, finesize):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize

        self.dir_A = os.path.join(self.root, self.phase + '/testA')
        self.dir_C = os.path.join(self.root, self.phase + '/testB')
        self.dir_D = os.path.join(self.root, self.phase + '/candidate')

        # image path
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.C_paths = sorted(make_dataset(self.dir_C))
        self.D_paths = sorted(make_dataset(self.dir_D))

        # transform
        self.transform = ToTensor()  # [0,1]

    def __getitem__(self, index):
        A = Image.open(self.A_paths[index]).convert("RGB")
        C = Image.open(self.C_paths[index]).convert("RGB")
        candidate = Image.open(self.D_paths[index]).convert('RGB')
        A = A.resize((self.fineSize, self.fineSize), Image.LANCZOS)
        C = C.resize((self.fineSize, self.fineSize), Image.LANCZOS)
        # strong augmentation
        strong_data_u = data_aug(A)
        strong_data_o = data_aug(C)
        tensor_w_u = self.transform(A)
        tensor_w_o = self.transform(C)
        tensor_s_u = self.transform(strong_data_u)
        tensor_s_o = self.transform(strong_data_o)
        tensor_d = self.transform(candidate)
        name = self.D_paths[index]

        return tensor_w_u, tensor_w_o, tensor_s_u, tensor_s_o, tensor_d, name

    def __len__(self):
        return len(self.A_paths)


class ValLabeled(data.Dataset):
    def __init__(self, dataroot, phase, finesize):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize

        self.dir_A = os.path.join(self.root, self.phase + '/trainA')
        self.dir_B = os.path.join(self.root, self.phase + '/trainB')
        self.dir_C = os.path.join(self.root, self.phase + '/trainC')

        # image path
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.C_paths = sorted(make_dataset(self.dir_C))

        # transform
        self.transform = ToTensor()  # [0,1]

    def __getitem__(self, index):
        # A, B is the image pair, hazy, gt respectively
        A = Image.open(self.A_paths[index]).convert("RGB")
        B = Image.open(self.B_paths[index]).convert("RGB")
        C = Image.open(self.C_paths[index]).convert("RGB")
        resized_a = A.resize((self.fineSize, self.fineSize), Image.LANCZOS)
        resized_b = B.resize((self.fineSize, self.fineSize), Image.LANCZOS)
        resized_c = C.resize((self.fineSize, self.fineSize), Image.LANCZOS)
        # transform to (0, 1)
        tensor_a = self.transform(resized_a)
        tensor_b = self.transform(resized_b)
        tensor_c = self.transform(resized_c)

        return tensor_a, tensor_b, tensor_c

    def __len__(self):
        return len(self.A_paths)


class TestData(data.Dataset):
    def __init__(self, dataroot, phase):
        super().__init__()
        self.root = dataroot
        self.phase = phase
        self.dir_A = os.path.join(self.root, self.phase + '/trainA')
        self.dir_C = os.path.join(self.root, self.phase + '/trainB')

        # image path
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.C_paths = sorted(make_dataset(self.dir_C))

        # transform
        self.transform = ToTensor()  # [0,1]

    def __getitem__(self, index):
        # A, B is the image pair, hazy, gt respectively
        A = Image.open(self.A_paths[index]).convert("RGB")
        C = Image.open(self.C_paths[index]).convert("RGB")
        name = self.A_paths[index].split('/')[-1]
        # transform to (0, 1)
        tensor_a = self.transform(A)
        tensor_c = self.transform(C)
        data_dict = {
            'under': tensor_a,
            'over': tensor_c,
            'name': name
        }

        return data_dict

    def __len__(self):
        return len(self.A_paths)


def data_aug(images):
    kernel_size = int(random.random() * 4.95)
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    blurring_image = transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
    strong_aug = images
    strong_aug = transforms.RandomGrayscale(p=0.2)(strong_aug)
    strong_aug = blurring_image(strong_aug)
    return strong_aug
