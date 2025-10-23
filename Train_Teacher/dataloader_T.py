# -- coding: utf-8 --
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import cv2
from augmentation import *
from utils import *
from Train_Teacher.Course import Course_Augmentation

transform_tensor = transforms.ToTensor()
# setup_logger(args.output)

def read_datasets(mode,args):
    images = []
    cell_labels = []
    nerve_labels = []

    if mode == "train":
        train_folder = os.path.join(args.data_path,'train')
        image_folder = os.path.join(train_folder, 'image')
        images_name = os.listdir(image_folder)  
        gt3_cell_folder = os.path.join(train_folder, 'cell_label')
        gt3_nerve_folder = os.path.join(train_folder, 'nerve_label')
             
    elif mode == "val":
        val_folder = os.path.join(args.data_path, "val")
        image_folder = os.path.join(val_folder, 'image')
        images_name = os.listdir(image_folder) 
        gt3_cell_folder = os.path.join(val_folder, 'cell_label')
        gt3_nerve_folder = os.path.join(val_folder, 'nerve_label')

    elif mode == "test":
        test_folder = os.path.join(args.data_path, "test")
        image_folder = os.path.join(test_folder, 'image')
        images_name = os.listdir(image_folder)    
        gt3_cell_folder = os.path.join(test_folder, 'cell_label')
        gt3_nerve_folder = os.path.join(test_folder, 'nerve_label')

    for name in images_name:
        img_path = os.path.join(image_folder, name)
        gt3_cell_path = os.path.join(gt3_cell_folder, name)
        gt3_nerve_path = os.path.join(gt3_nerve_folder, name)

        images.append(img_path)
        cell_labels.append(gt3_cell_path)
        nerve_labels.append(gt3_nerve_path)
     
    return images, nerve_labels, cell_labels, images_name 


class MyDataset(Dataset):
    def __init__(self,args,mode="train"):
        self.mode = mode
        self.images, self.nerve_labels, self.cell_labels, self.images_name = read_datasets(self.mode,args)
        self.args = args

    def __getitem__(self, index):

        image_path = self.images[index]
        nerve_label_path = self.nerve_labels[index]
        cell_label_path = self.cell_labels[index]
        image_name = self.images_name[index]
    
        # Image is (H,W,C),label is (H,W,1)
        image = cv2.imread(image_path) #even if png format, its shape is still (H,W,3)
        nerve_label = cv2.imread(nerve_label_path,0)
        cell_label = cv2.imread(cell_label_path,0)

        if len(image.shape) == 2: # (H,W,C)
            image = image[:,:,np.newaxis] 
            image = np.repeat(image,3,axis=-1) #（H,W,3）

        nerve_label = nerve_label[:,:,np.newaxis] #(H,W,1)
        cell_label = cell_label[:,:,np.newaxis] #(H,W,1)

        if self.args.crop == True and self.mode == "train":
            # image,label,label = center_crop_and_pad(image, label, label, self.args.img_size[0], self.args.img_size[1])
            image,nerve_label,cell_label = crop_images_and_label(image, nerve_label, cell_label, self.args.roi_size) # image:[H,W,C]  label:[H,W,1] random_label:[H,W,1]
        
        if self.args.train_SDF == False and self.mode == self.args.enhance_mode_T:
            image, nerve_label, cell_label = apply_augmentations_KD(image,nerve_label,cell_label)
 
        label = cv2.add(nerve_label,cell_label) # (H,W,1) 
    
        # beta = round(random.uniform(0.1,10),2)  
        if self.args.data_path_selection == "teacher" and self.args.enhance_mode_T == "train" and self.args.train_SDF == True:   
            beta = random.randint(1, 6)  
            SDF_image = Course_Augmentation(image, label, beta, self.args.Course_inference_folder_path)

        else:  
            SDF_image,_ = get_SDF_data(image,label,self.args.beta) 
            # SDF_image = image
    
        # label = cv2.add(nerve_label,cell_label) # (H,W,1)  
        # dialate_pixels = random.randint(10,150)
        # SDF_image = dialated_plain(image,label,dialate_pixels)
        
        # SDF_image = deepcopy(image)
        
        image = transform_tensor(image) # supply test's droped cells
        SDF_image = transform_tensor(SDF_image) # supply test's droped cells
        nerve_label = transform_tensor(nerve_label) #(4,240,240) NO automatic normalization!!!
        cell_label = transform_tensor(cell_label) #(4,240,240) NO automatic normalization!!!

        return image,SDF_image,nerve_label,cell_label,image_name

    def __len__(self):
        
        assert len(self.images) == len(self.nerve_labels) == len(self.cell_labels), "The length of images, labels and names should be equal!"   
        return len(self.images)


class Data_loader(): 
    def __init__(self):
        pass

    def load_train_data(self, args,batch_size):
        dataset = MyDataset(args,mode="train")
        train_loader = DataLoader(dataset, batch_size, shuffle=True, pin_memory=False)
        return train_loader
    
    def load_val_data(self,args,batch_size):
        dataset = MyDataset(args,mode="val")
        val_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True, pin_memory=False)
        return val_loader
    
    def load_test_data(self,args,batch_size):
        dataset = MyDataset(args,mode="test")
        test_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True, pin_memory=False)
        return test_loader
    
