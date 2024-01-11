import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import os.path
import torchvision.transforms as transforms
import PIL.Image as Image
# from core.utils import frame_utils
# from utils.augmentor import FlowAugmentor, SparseFlowAugmentor

class VimeoTripletsDataset(data.Dataset):
    def __init__(self, root="/home/jyzhao/Code/Datasets/vimeo_triplet/sequences"):
        self.root = root
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.augmentor = None
        self.is_test = False
        self.init_seed = False
        self.image_list = []
        self.extra_info = []
        self.sequence_folders = []
        #self.sequence_folders = [os.path.join(root, f) for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
                # Iterate through all folders and subfolders to collect image paths
        for folder in sorted(os.listdir(root)):
            folder_path = os.path.join(root, folder)
            for subfolder in sorted(os.listdir(folder_path)):
                # subfolder_path = os.path.join(folder_path, subfolder)
                self.sequence_folders.append(os.path.join(folder_path, subfolder))
                # for image_file in sorted(os.listdir(subfolder_path)):
                #     self.sequence_folders.append(os.path.join(subfolder_path, image_file))

    def __len__(self):
        return len(self.sequence_folders)
    
    def crop(self, img1, img2, img3, h, w):
        ih, iw, _ = img1.shape # 3, 256, 448 --> 256, 448, 3
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img1 = img1[x:x+h, y:y+w, :]
        img2 = img2[x:x+h, y:y+w, :]
        img3 = img3[x:x+h, y:y+w, :] # 224,224,3
        return img1, img2, img3
        
    def __getitem__(self, index):
        sequence_folder = self.sequence_folders[index]
        img1 = Image.open(os.path.join(sequence_folder, 'im1.png'))
        img2 = Image.open(os.path.join(sequence_folder, 'im2.png'))
        img3 = Image.open(os.path.join(sequence_folder, 'im3.png')) # .permute(2, 0, 1)

        #if self.transform:
        img1 = self.transform(img1).permute(1, 2, 0)
        img2 = self.transform(img2).permute(1, 2, 0)
        img3 = self.transform(img3).permute(1, 2, 0)

        
        img1, img2, img3 = self.crop(img1, img2, img3, 224, 224)
        img1 = img1.permute(2,0,1)
        img2 = img2.permute(2,0,1)
        img3 = img3.permute(2,0,1)
        return torch.cat((img1, img2, img3), 0)
        pass

        

        return returnit
def triplet_dataloader(args):
    dataset = VimeoTripletsDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, pin_memory=False, shuffle=True, num_workers=4, drop_last=True)
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, pin_memory=False, shuffle=True, num_workers=4, drop_last=True)
    
    print('Training with %d image pairs' % len(dataset))

    return dataloader




