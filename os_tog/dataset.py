
import os
import cv2
import random
import numpy as np
from skimage import measure
from PIL import Image
from PIL.Image import Transform, Transpose
import json
from tqdm import tqdm
import torch
import torchvision
from torch.utils.data import Dataset
from pycocotools import mask as pycoco_mask
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import BatchSampler
import torch.nn as nn
import torchvision


#############################################################################
# ------------------------ OBJECT RECOGNITION ------------------------- #
#############################################################################

class GuidedEmbeddingNet(nn.Module):
    def __init__(self):
        # pre-trained resnet-50 on ImageNet for extracting product features
        super(GuidedEmbeddingNet, self).__init__()
        # load resnet50 pre-trained on ImageNet
        self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)

        # remove layers
        to_remove = -1 # number of layers to remove (-1 = 1, -2 = 2)
        self.model = nn.Sequential(*list(self.model.children())[:to_remove])
    
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.normalize(x, p=2)
        return x
    
class ObjectRecognitionDataset(Dataset):
    def __init__(self, root_dir, split="training", transforms=None, ref_transforms=None, annotation_file=None, GE=False, RP=False, ADD_MASK=False) -> None:
        """ Create a custom PyTorch dataset class for object recognition. This works for single-object.
        
        @param root_dir: (str) path to root of dataset directory.
        @param split: (str) whether the dataset is for 'training' or 'testing'.
        @param transforms: (Optional) transformations for images.
        @param annotation_file: (str) path to annotation file, if left at None user will be prompted to generate one.
        """
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        self.ref_transforms = ref_transforms

        self.GUIDED_EMBEDDINGS = GE
        self.RANDOM_PROD = RP
        self.ADD_MASK = ADD_MASK

        # load annotations and generate image info and class info
        self.load_annotations(annotation_file)

    def load_ref_img(self, ref_idx, transform=True): 
        # return img and class
        anchor_img = Image.open(self.ref_image_info[ref_idx]['path']).convert("RGB")
        anchor_cls = self.ref_image_info[ref_idx]['category_id']
        if transform:
            anchor_img = self.ref_transforms(anchor_img)
        return anchor_img, anchor_cls

    def load_annotations(self, annotation_file, return_annots=False):
        """ Loads the JSON annotation object and generates dictionaries for image info and class info. 
        Note - if you want to exclude any object classes from the dataset this is where it's best to set it up. 

        @param annotation_file: (str) path to annotation file.
        @param return_annots: (bool) whether to return the JSON annotation object.
        """
        # load JSON object
        f = open(annotation_file)
        annotations = json.load(f)

        # store image info (scene images) and class info
        self.image_info, self.class_info = {}, {}
        
        # generate class info
        for i in annotations['categories']:
            self.class_info[i['id']] =  i['name']

        # create mappings
        self.obj_img_map = {i:[] for i in self.class_info.keys()} # keys are object category idx, values are lists of image idxs
        self.img_obj_map = {} # keys are img idx, values are object category idxs

        for i in annotations['images']:  # generate image info
            self.image_info[i['id']] = {
                'path': os.path.join(self.root_dir, i['file_name']),
                'width': i['width'],
                'height': i['height'],
                'annotations': [v for v in annotations['annotations'] if v['image_id'] == i['id']]
            }
            # if its not multi-object there will only be one annotation per image so we can just take [0]
            object_idx = self.image_info[i['id']]['annotations'][0]['category_id']
            self.img_obj_map[i['id']] = object_idx # each image has one object class only in single-object scenes
            self.obj_img_map[object_idx].append(i['id']) 

        # create reference image info if provided - otherwise take the first img idx in obj_img_map
        self.ref_image_info = {}
        self.obj_ref_map = {i:[] for i in self.class_info.keys()} # keys are object category idx, values are lists of ref image idxs
        if 'reference_images' in annotations:
            for i in annotations['reference_images']:
                self.ref_image_info[i['id']] = {
                'path': os.path.join(self.root_dir, i['file_name']),
                'width': i['width'],
                'height': i['height'],
                'category_id': i['category_id']
                }
                self.obj_ref_map[i['category_id']].append(i['id'])
        else:
            for obj_idx in self.obj_ref_map.keys(): # only tested for UMD
                # get the number of affordances for each img idx
                affordances = [len([aff['aff_id'] for ann in self.image_info[img_idx]["annotations"] for aff in ann['affordances']]) for img_idx in  self.obj_img_map[obj_idx]]
                max_aff_ref_idx = np.argmax(affordances)
                # get the first reference image with the highest number of affordances instead of just first
                img_idx = self.obj_img_map[obj_idx][max_aff_ref_idx] 
                self.ref_image_info[img_idx] = {
                    'path': self.image_info[img_idx]['path'],
                    'width': self.image_info[img_idx]['width'],
                    'height': self.image_info[img_idx]['height'],
                    'category_id': obj_idx
                }
                self.obj_ref_map[obj_idx].append(img_idx)

        if self.GUIDED_EMBEDDINGS:
            print("[INFO] Generating guided embeddings for each product image class...")
            # get guided embeddings beforehand, by object class id
            model = GuidedEmbeddingNet()
            DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            model.to(DEVICE)
            model.eval()
            self.prod_img_features = {} # dictionary of product features, indexed by object class id
            for object_idx in tqdm(self.class_info.keys()):
                self.prod_img_features[object_idx] = [] # create empty list to store features of one object class
                for prod_img_idx in self.obj_ref_map[object_idx]:
                    # load and pre-process product image
                    prod_img = Image.open(self.ref_image_info[prod_img_idx]['path']).convert("RGB")
                    if self.ADD_MASK:
                        anch_mask, _ = self.load_mask(prod_img_idx) # prod_img_idx should be same idx as in image_info see else statement above
                        prod_img = self.apply_mask(prod_img, anch_mask)
                    prod_img = self.ref_transforms(prod_img)[None, :, :, :] # expand dimension to include batch
                    prod_img = prod_img.to(DEVICE) # put on GPU
                    # extract features from product image using pre-trained model
                    with torch.no_grad():
                        prod_feature = model(prod_img)
                    self.prod_img_features[object_idx].append(prod_feature.cpu().detach().numpy()[0])
                self.prod_img_features[object_idx] = np.array(self.prod_img_features[object_idx])
            print("[INFO] Finished generating guided embeddings.")

        if return_annots:
            return annotations
    
    def __getitem__(self, idx):
        image_id = idx + 1 # add 1 because our image idxs start at 1 not 0 in the annotations (because standard pytorch dataset object goes from idx 0 - len(dataset))

        # get training image (480, 640, 3)
        positive_label = self.img_obj_map[image_id]
        positive_img = Image.open(self.image_info[image_id]['path']).convert("RGB")

        # get a random training image with different object class (negative example)
        negative_label = random.choice([l for l in self.class_info.keys() if l != positive_label])
        neg_image_id = random.choice(self.obj_img_map[negative_label])
        negative_img = Image.open(self.image_info[neg_image_id]['path']).convert("RGB")


        if self.GUIDED_EMBEDDINGS:
            # select anchor image (i.e. reference image) - first one by default
            anchor_img = self.prod_img_features[positive_label][0]
        elif self.RANDOM_PROD:
            # select anchor image (i.e. reference image) - first one by default
            anchor_image_id = random.choice(self.obj_ref_map[positive_label])
            anchor_img = Image.open(self.ref_image_info[anchor_image_id]['path']).convert("RGB")
            if self.ADD_MASK:
                anch_mask, _ = self.load_mask(anchor_image_id) # single-object so get one mask
                anchor_img = self.apply_mask(anchor_img, anch_mask)
        else:
            # select anchor image (i.e. reference image) - first one by default
            anchor_img = Image.open(self.ref_image_info[self.obj_ref_map[positive_label][0]]['path']).convert("RGB")
            if self.ADD_MASK:
                anch_mask, _ = self.load_mask(self.obj_ref_map[positive_label][0]) # single-object so get one mask
                anchor_img = self.apply_mask(anchor_img, anch_mask)

        if self.ADD_MASK: # only tested for UMD
            pos_mask, _ = self.load_mask(image_id) # single-object so get one mask
            positive_img = self.apply_mask(positive_img, pos_mask)
            neg_mask, _ = self.load_mask(neg_image_id)
            negative_img = self.apply_mask(negative_img, neg_mask)

        if self.transforms:
            positive_img = self.transforms(positive_img)
            negative_img = self.transforms(negative_img)
        if self.ref_transforms and not self.GUIDED_EMBEDDINGS: # don't need pre-processing if using embeddings - was pre-processed in load_annotations
            anchor_img = self.ref_transforms(anchor_img)

        sample = {"anchor_img": anchor_img, "pos_img":positive_img, "neg_img":negative_img, "pos_label":positive_label, "neg_label":negative_label}
        return sample

    def apply_mask(self, image, mask):
        np_img = np.asarray(image) # [480,640,3]
        mask = np.repeat(mask[0][:,:,None], 3, axis=-1)
        np_img = np_img * mask
        new_img = Image.fromarray(np.uint8(np_img))
        return new_img

    def __len__(self):
        """ Returns the number of images in the dataset. """
        return len(self.image_info.keys())
    
    def load_mask(self, image_id):
        image_info = self.image_info[image_id]

        instance_masks = []
        class_ids = [] # TODO consider removing this from here (don't need class IDs)

        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = annotation["category_id"]
            m = self.annToMask(annotation, image_info["height"],
                                image_info["width"])
            # Some objects are so small that they're less than 1 pixel area
            # and end up rounded out. Skip those objects.
            if m.max() < 1:
                continue
            instance_masks.append(m)
            class_ids.append(class_id)

        mask = np.stack(instance_masks, axis=0).astype(np.int32)
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids
    
    def annToRLE(self, ann, height, width):
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts we merge all parts into one mask rle code
            rles = pycoco_mask.frPyObjects(segm, height, width)
            rle = pycoco_mask.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = pycoco_mask.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        rle = self.annToRLE(ann, height, width)
        m = pycoco_mask.decode(rle)
        return m
    
    def generate_references(self, model):
        DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # FIRST - we get features of all product/reference images
        prodImgFeats = np.zeros((len(self.ref_image_info.keys()), 2048))
        prodImgClass = np.zeros((len(self.ref_image_info.keys())))
        prodImgIdxs = np.zeros((len(self.ref_image_info.keys())))

        print("[INFO] Generating product image features...")
        pbar = tqdm(self.ref_image_info.items())
        for i, sample in enumerate(pbar):
            ref_idx, ref = sample
            img, cls = self.load_ref_img(ref_idx, transform=False)
            anch_mask, _ = self.load_mask(ref_idx) # prod_img_idx should be same idx as in image_info see else statement above
            img = self.apply_mask(img, anch_mask)
            img = self.ref_transforms(img)
            processed_img = img[None, :, :, :]
            processed_img = processed_img.to(DEVICE)
            with torch.no_grad():
                feature = model(processed_img)
            prodImgFeats[i] = feature.cpu().detach().numpy()
            prodImgClass[i] = cls
            prodImgIdxs[i] = ref_idx
        return prodImgFeats, prodImgClass, prodImgIdxs


class ObjectRecognitionBatchSampler(BatchSampler):
    """ Balanced batch sampler that samples X samples of N classes in each batch. 
    Modified from https://github.com/adambielski/siamese-triplet/blob/master/datasets.py """

    def __init__(self, dataset, n_classes, n_samples):
        obj_img_map = dataset.obj_img_map
        self.labels_set = list(obj_img_map.keys())
        self.label_to_indices = {label: [i-1 for i in obj_img_map[label]] # subtract 1 to get dataset idx instead of img idx
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(dataset)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


#############################################################################
# ------------------------ AFFORDANCE RECOGNITION ------------------------- #
#############################################################################

class AffordanceRecognitionDataset(Dataset):
    def __init__(self, root_dir, transforms=None, annotation_file=None, add_mask=False) -> None:
        """ Create a custom PyTorch dataset class for affordance recognition.
        
        @param root_dir: (str) path to root of dataset directory.
        @param transforms: (Optional) transformations for images.
        @param annotation_file: (str) path to annotation file, if left at None user will be prompted to generate one.
        @param add_mask: (bool) whether to add object mask to image. Only works for single-object scenes for now.
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.add_mask = add_mask
        
        # load annotations and generate image info and class info
        self.load_annotations(annotation_file)

    def load_annotations(self, annotation_file, return_annots=False):
        """ Loads the JSON annotation object and generates dictionaries for image info and class info. 
        Note - if you want to exclude any object classes from the dataset this is where it's best to set it up. 

        @param annotation_file: (str) path to annotation file.
        @param return_annots: (bool) whether to return the JSON annotation object.
        """
        # load JSON object
        f = open(annotation_file)
        annotations = json.load(f)

        # store image info and class info, affordance info in dictionaries
        self.image_info, self.class_info, self.affordance_info = {}, {}, {}

        # generate affordance info
        for i in annotations['affordances']:
            self.affordance_info[i['id']] = i['name'] 
        
        # generate class info
        for i in annotations['categories']:
            self.class_info[i['id']] =  i['name']

        # create mappings
        self.aff_obj_map = {i:[] for i in self.affordance_info.keys()} # keys are affordance idx, values are lists of object category idxs
        self.obj_img_map = {i:[] for i in self.class_info.keys()} # keys are object category idx, values are lists of image idxs
        self.obj_img_affn_map = {i:[] for i in self.class_info.keys()} # get the total number of affordances each img_idx has
        self.img_obj_map = {} # keys are img idx, values are object category idxs

        for i in annotations['images']:  # generate image info
            self.image_info[i['id']] = {
                'path': os.path.join(self.root_dir, i['file_name']),
                'width': i['width'],
                'height': i['height'],
                'annotations': [v for v in annotations['annotations'] if v['image_id'] == i['id']]
            }
            # map each image to an object class ID and vice versa
            classes = [a['category_id'] for a in self.image_info[i['id']]['annotations']]
            affs = [[aff['aff_id'] for aff in ann['affordances']] for ann in self.image_info[i['id']]['annotations']] # get affordances for each object
            self.img_obj_map[i['id']] = classes # TODO - think if you want repeats in classes list?
            for y, obj_idx in enumerate(classes):
                self.obj_img_map[obj_idx].append(i['id']) 
                # get number of affordances for this image
                self.obj_img_affn_map[obj_idx].append(len([aff['aff_id'] for ann in self.image_info[i['id']]["annotations"] for aff in ann['affordances']]))
                # map affordances for class ids
                for aff_idx in affs[y]:
                    if classes[y] not in self.aff_obj_map[aff_idx]:
                        self.aff_obj_map[aff_idx].append(classes[y])

        if return_annots:
            return annotations
    
    def __getitem__(self, idx):
        """ Optional - aff_idx to get a specific affordance with that id. """
        image_id = idx + 1 # add 1 because our image idxs start at 1 not 0

        # load image and masks
        img = Image.open(self.image_info[image_id]['path']).convert("RGB")
        masks, class_ids = self.load_mask(image_id)
        aff_masks, aff_class_ids = self.load_affs(image_id)

        # get bounding boxes from [x_min, y_min, w, h] to [x_min, y_min, x_max, y_max]
        bboxes = [a['bbox'] for a in self.image_info[image_id]['annotations']]
        boxes = []
        for b in bboxes:
            boxes.append([b[0], b[1], b[0] + b[2], b[1] + b[3]])
        boxes = np.array(boxes)

        # convert everything to tensors
        image_id = torch.tensor([image_id])
        labels = torch.as_tensor(class_ids, dtype=torch.int64)
        aff_labels = torch.as_tensor( aff_class_ids, dtype=torch.int64)

        target = {}
        target["labels"] = labels # [N]
        target["affs_labels"] = aff_labels # [N, A]
        target["image_id"] = image_id # [1]
        target["masks"] = masks # [N, H, W]
        target["boxes"] = boxes # [N, [x_min, y_min, x_max, y_max]]
        target["affs"] = aff_masks # [N, A, H, W]

        if self.add_mask:
            img = self.apply_mask(img, masks) # mask out background

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        """ Returns the number of images in the dataset. """
        return len(self.image_info.keys())
    
    def load_reference_image(self, obj_id):
        max_aff_ref_idx = np.argmax(self.obj_img_affn_map[obj_id]) # which img idx for that object id has the most affordances
        return self.__getitem__(self.obj_img_map[obj_id][max_aff_ref_idx]-1) # load that img_idx

    def get_reference_affordance_from_image(self, ref_t, aff_name):
        aff_id = [k for k, v in self.affordance_info.items() if v==aff_name][0]
        aff_idx = np.where(ref_t["affs_labels"][0] == aff_id)[0]
        aff_mask = ref_t["affs"][0][aff_idx[0]]
        return aff_mask, aff_id
    
    def apply_mask(self, image, mask):
        """ Applies object mask on image. Note this will only work in single-object scenes for now. """
        np_img = np.asarray(image) # [480,640,3]
        mask = np.repeat(mask[0][:,:,None], 3, axis=-1)
        np_img = np_img * mask
        new_img = Image.fromarray(np.uint8(np_img))
        return new_img

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [instances, height, width].

        Returns:
        masks: A bool array of shape [instance count, height, width] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]

        instance_masks = []
        class_ids = [] # TODO consider removing this from here (don't need class IDs)

        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = annotation["category_id"]
            m = self.annToMask(annotation, image_info["height"],
                                image_info["width"])
            # Some objects are so small that they're less than 1 pixel area
            # and end up rounded out. Skip those objects.
            if m.max() < 1:
                continue
            instance_masks.append(m)
            class_ids.append(class_id)

        mask = np.stack(instance_masks, axis=0).astype(np.bool)
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids
    
    def load_affs(self, image_id):
        image_info = self.image_info[image_id]

        aff_masks = [] # list of affordance masks
        aff_ids = [] # list of affordance ids for each object

        annotations = self.image_info[image_id]["annotations"]
        for annotation in annotations:
            affs = annotation['affordances']
            m = [self.annToMask(aff, image_info["height"],
                                image_info["width"]) for aff in affs]
            c = [aff['aff_id'] for aff in affs]
            aff_masks.append(m)
            aff_ids.append(c)

        mask = np.stack(aff_masks, axis=0).astype(np.bool)
        aff_ids = np.array(aff_ids, dtype=np.int32)
        return mask, aff_ids

    def annToRLE(self, ann, height, width):
        """ Convert annotation which can be polygons, uncompressed RLE to RLE.
        @return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = pycoco_mask.frPyObjects(segm, height, width)
            rle = pycoco_mask.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = pycoco_mask.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """ Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        @return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = pycoco_mask.decode(rle)
        return m
    

#############################################################################
# ---------------------------- GRASP DETECTION ---------------------------- #
#############################################################################

class GraspDetectionDataset(Dataset):
    def __init__(self, root_dir, split="training", transforms=None,  annotation_file=None, rot_classes=12, multi_object=False, mask_transforms=None) -> None:
        """ Create a custom PyTorch dataset class for grasp detection.
        
        @param root_dir: (str) path to root of dataset directory.
        @param split: (str) whether the dataset is for 'training' or 'validation'.
        @param annotation_file: (str) path to annotation file COCO-style format.
        @param transforms: (Compose, Optional) transformations for images.
        @param rot_classes: (int) number of rotation classes to use.
        """
        self.root_dir = root_dir
        self.split = split
        self.multi_object = multi_object
        self.transforms = transforms
        self.mask_transforms = mask_transforms
        self.rot_classes = rot_classes

        # load annotations and generate image info and class info
        self.load_annotations(annotation_file)
    
    def load_annotations(self, annotation_file, return_annots=False):
        """ Loads the JSON annotation object and generates dictionaries for image info and class info. 
        Note - if you want to exclude any object classes from the dataset this is where it's best to set it up. 
        
        @param annotation_file: (str) path to annotation file.
        @param return_annots: (bool) whether to return the JSON annotation object.
        """
        # load JSON object
        f = open(annotation_file)
        annotations = json.load(f)

        # store image info and class info into two dictionaries
        self.image_info, self.class_info = {}, {}

        # generate class info (we only have two classes)
        self.class_values = self.generate_rotation_classes() # maps class ids to [min, max] theta values
        self.class_info = {0: 'background'}
        for key in self.class_values.keys():
            self.class_info[key] = f"{int(self.class_values[key][0]*(180/np.pi))} >= \u03B8 < {int(self.class_values[key][1]*(180/np.pi))}"

        # TODO - make 'annotations' only store necessary fields (e.g. bounding boxes, image classes not needed here)
        for i in annotations['images']:  # generate image info
            self.image_info[i['id']] = {
                'path': os.path.join(self.root_dir, i['file_name']),
                'width': i['width'],
                'height': i['height'],
                'annotations': [v for v in annotations['annotations'] if v['image_id'] == i['id']]
            }

        if return_annots:
            return annotations
        
    def __getitem__(self, idx):
        image_id = idx + 1 # add 1 because our image idxs start at 1 not 0

        # load image and masks
        img = Image.open(self.image_info[image_id]['path']).convert("RGB")

        # get all grasps for this image
        grasps = []
        if self.split == "testing" and self.multi_object: # for evaluating in multi-object scenes
            obj_cls = [] # object classes of each grasp (not really important except for visualisation)
            unique_obj_cls = [] # unique ID to differentiate object clases of same type
            obj_masks = {} # object masks of each grasp (unique obj to mask)
            for i, a in enumerate(self.image_info[image_id]['annotations']):
                obj_cls += [a['category_id']] * len(a['grasps'])
                unique_obj_cls += [i] * len(a['grasps'])
                obj_mask = np.asarray(self.annToMask(a, self.image_info[image_id]["height"], self.image_info[image_id]["width"]), dtype=np.bool)
                if self.mask_transforms is not None:
                    obj_mask = self.mask_transforms(obj_mask)
                obj_masks[i] = obj_mask
                grasps += a['grasps']
            obj_cls = np.array(obj_cls, dtype=np.int32)
            unique_obj_cls = np.array(unique_obj_cls, dtype=np.int32)
        else:
            for a in self.image_info[image_id]['annotations']:
                grasps += a['grasps']
        num_objs = len(grasps)
        grasps = np.array(grasps) # convert to array before transformations

        # transform image and grasps
        target = {"grasps": grasps}
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        grasps = target["grasps"]

        # now we convert the transformed grasps into the expected format for object detector (box + label)
        boxes, labels = [], []
        for i in grasps:
            xmin = i[0] - (i[2] / 2)
            xmax = i[0] + (i[2] / 2)
            ymin = i[1] - (i[3] / 2)
            ymax = i[1] + (i[3] / 2)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.theta_to_class(i[4]))
        
        # convert everything to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([image_id])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes # [N, [x_min, y_min, x_max, y_max]]
        target["labels"] = labels # [N]
        target["image_id"] = image_id # [1]
        target["area"] = area # [N]
        target["iscrowd"] = iscrowd # [N]
        
        if self.split == "testing" and self.multi_object:
            object_info = {
                "obj_cls": obj_cls,
                "unique_obj_cls": unique_obj_cls,
                "obj_masks": obj_masks
            }
            return img, target, object_info
        else:
            return img, target
    
    def __len__(self):
        """ Returns the number of images in the dataset. """
        return len(self.image_info.keys())
    
    def generate_rotation_classes(self):
        """ Returns a dictionary that maps theta (i.e. rotation) values to a suitable rotation idx. Note that the idx 0
        represents an invalid grasp. Note the minimum theta value can be -pi/2 and the maximum theta value can be pi/2.
           :return class_list: (dict) dictionary where idxs are mapped to unique rotation intervals. Each item in the
           dictionary has the form 'idx':[min_theta_val, max_theta_val]. Hence, a theta value that is >= "min_theta_val"
           and < "max_theta_val" of one of the rotation classes is assigned the "idx" of that particular class.
        """
        class_list = {}
        rot_range = 180  # the range of rotation values
        rot_start = -np.pi / 2
        rot_size = (rot_range / self.rot_classes) * (np.pi / 180)  # diff. in rotation between each idx (radians)
        for i in range(self.rot_classes):
            min_rot, max_rot = rot_start + (i * rot_size), rot_start + ((i + 1) * rot_size)
            class_list[i + 1] = [min_rot, max_rot]
        return class_list
    
    def theta_to_class(self, theta):
        """ Assigns a given rotation value (in radians) to a suitable rotation class idx.
          :param theta: (float) a rotation value in radians.
          :return: (int) an assigned rotation class.
       """
        for key, value in self.class_values.items():
            if value[0] <= theta < value[1]:
                return key
        return self.rot_classes
    
    def class_to_theta(self, cls):
        """ Get the median theta values of a rotation class. """
        return (self.class_values[cls][0] + self.class_values[cls][1]) / 2
    
    def VOC_to_5D_grasp_pose(self, bbox, cls):
        """ Given a bbox in VOC format [x_min, y_min, x_max, y_max] and rotation class, get a 5D grasp pose (cx, cy, w, h, t). """
        xmin, ymin, xmax, ymax = bbox
        w, h = xmax - xmin, ymax - ymin
        x, y = xmax - (w / 2), ymax - (h / 2)
        t = self.class_to_theta(cls)
        return x, y, w, h, t
    
    def annToRLE(self, ann, height, width):
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = pycoco_mask.frPyObjects(segm, height, width)
            rle = pycoco_mask.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = pycoco_mask.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        rle = self.annToRLE(ann, height, width)
        m = pycoco_mask.decode(rle)
        return m


#############################################################################
# ------------------------- INSTANCE SEGMENTATION ------------------------- #
#############################################################################

class InstanceSegmentationDataset(Dataset):
    def __init__(self, root_dir, split="training", transforms=None, annotation_file=None):
        """ Create a custom PyTorch dataset class for instance segmentation.
        
        @param root_dir: (str) path to root of dataset directory.
        @param split: (str) whether the dataset is for 'training' or 'validation'.
        @param annotation_file: (str) path to annotation file, if left at None user will be prompted to generate one.
        @param transforms: (Optional) transformations for images.
        """
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        
        # load annotations and generate image info and class info
        self.load_annotations(annotation_file)

    def load_annotations(self, annotation_file, return_annots=False):
        """ Loads the JSON annotation object and generates dictionaries for image info and class info. 
        If you want to exclude any object classes from the dataset this is where it's best to set it up. 
        
        @param annotation_file: (str) path to annotation file.
        @param return_annots: (bool) whether to return the JSON annotation object.
        """
        # load JSON object
        f = open(annotation_file)
        annotations = json.load(f)

        # store image info and class info into two dictionaries
        self.image_info, self.class_info = {}, {}

        # generate class info (we only have two classes)
        self.class_info = {
            0: 'background',
            1: 'object'
        }

        # TODO - make 'annotations' only store necessary fields (e.g. grasps not needed here)
        for i in annotations['images']:  # generate image info
            self.image_info[i['id']] = {
                'path': os.path.join(self.root_dir, i['file_name']),
                'depth_path': os.path.join(self.root_dir, i['depth_name']),
                'width': i['width'],
                'height': i['height'],
                'annotations': [v for v in annotations['annotations'] if v['image_id'] == i['id']]
            }

        if return_annots:
            return annotations
    
    def __getitem__(self, idx):
        image_id = idx + 1 # add 1 because our image idxs start at 1 not 0

        # load image and masks
        img = Image.open(self.image_info[image_id]['path']).convert("RGB")
        masks, _ = self.load_mask(image_id)

        # get bounding boxes from [x_min, y_min, w, h] to [x_min, y_min, x_max, y_max]
        bboxes = [a['bbox'] for a in self.image_info[image_id]['annotations']]
        num_objs = len(bboxes)
        boxes = []
        for b in bboxes:
            boxes.append([b[0], b[1], b[0] + b[2], b[1] + b[3]])
        boxes = np.array(boxes)

        # convert everything to tensors
        image_id = torch.tensor([image_id])
        # IMPORTANT - we will only have two classes - background and object
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # calculate area and assume all instances are not iscrowd
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes # [N, [x_min, y_min, x_max, y_max]]
        target["labels"] = labels # [N]
        target["masks"] = masks # [N, H, W]
        target["image_id"] = image_id # [1]
        target["area"] = torch.as_tensor(area, dtype=torch.float32) # [N]
        target["iscrowd"] = iscrowd # [N]

        if self.transforms is not None:
            img, target = self.transforms(img, target)
       
        return img, target

    def __len__(self):
        """ Returns the number of images in the dataset. """
        return len(self.image_info.keys())
    
    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [instances, height, width].

        Returns:
        masks: A bool array of shape [instance count, height, width] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]

        instance_masks = []
        class_ids = [] # TODO consider removing this from here (don't need class IDs)

        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = annotation["category_id"]
            m = self.annToMask(annotation, image_info["height"],
                                image_info["width"])
            # Some objects are so small that they're less than 1 pixel area
            # and end up rounded out. Skip those objects.
            if m.max() < 1:
                continue
            instance_masks.append(m)
            class_ids.append(class_id)

        mask = np.stack(instance_masks, axis=0).astype(np.bool)
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids

    def annToRLE(self, ann, height, width):
        """ Convert annotation which can be polygons, uncompressed RLE to RLE.
        @return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = pycoco_mask.frPyObjects(segm, height, width)
            rle = pycoco_mask.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = pycoco_mask.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """ Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        @return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = pycoco_mask.decode(rle)
        return m


#############################################################################
# ------------------------------- UMD DATASETS ---------------------------- #
#############################################################################

def generate_COCO_from_UMD(dataset_path, split="training", stype="category", save_path=None):
    """ Produces a COCO formatted JSON file from UMD dataset and saves it in the dataset's root directory.

    @param dataset_path: (str) path to OCID grasp dataset root directory.
    @param split: (str) whether to generate annotations for the 'training' or 'testing' data splits provided.
    @param save_path: (str, Optional) path where to save JSON file, if left at None the file will be saved
                      in the 'dataset_path' folder.

    Note we add 'affordances' and 'super_categories' here.
    """
    print(f'[INFO] Creating "{split}" annotations for the UMD dataset (dataset_path: "{dataset_path}") ...') 
    annot = {'info':{}, 'licenses':[], 'images':[], 'categories':[], 'super_categories':[], 'affordances':[], 'annotations':[]} 

    # -----------------  set info parameters -----------------
    annot['info']['decription'] = "UMD Part-Affordance Dataset"
    annot['info']['url'] = "http://users.umiacs.umd.edu/~fer/affordance/part-affordance-dataset/"
    annot['info']['version'] = "1.0"
    annot['info']['year'] = 2023

    # paths
    category_path = os.path.join(dataset_path, "tool_categories.txt")
    split_path = os.path.join(dataset_path, f"{stype}_split.txt")
    img_dir_path = os.path.join(dataset_path, "tools")

    # get the classes included the chosen data split (train, test, object, novel etc.)
    split_lines = read_lines_from_text_file(split_path)
    keep_id = '1' # 1 for training split and 2 for testing split
    if split == "testing":
        keep_id = '2' 
    objects_to_keep = [line.split()[1] for line in split_lines if line.split()[0] == keep_id] # names of object classes to keep
    print(f"[INFO] Object classes to use: {len(objects_to_keep)}")

    # ------------------ set affordances --------------------
    affordance_map = {1:'grasp', 2:'cut', 3:'scoop', 4:'contain', 5:'pound', 6:'support', 7:'wrap-grasp'}
    for i, name in affordance_map.items():
        annot['affordances'].append({
            'id': i,
            'name': name
        })

    # -------------------  set object categories ---------------------
    category_lines = read_lines_from_text_file(category_path)
    for i, line in enumerate(category_lines):
        x = line.split()
        idx, name = x[0], x[1]
        super_name = name.split("_")[0] # super category name
        
        # check if super-category exists
        if not any(s['name'] == super_name for s in annot['super_categories']):
            annot['super_categories'].append({
                'id': idx,
                'name': super_name
            })

        # check if category exists
        if name in objects_to_keep:  # only keep included objects in this split
            category = {
                'id': int(i) + 1,
                'name': name,
                'supercategory': name.split("_")[0]
            }
            annot['categories'].append(category)

    # ---------------- set images and annotations ---------------
    # get total number of instances to traverse
    total_files = [(obj, file) for obj in objects_to_keep for file in os.listdir(os.path.join(img_dir_path, obj)) if file.endswith("_rgb.jpg")]
    pbar = tqdm(total_files, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')
    for i, l in enumerate(pbar): # loops through each file path
        obj_dir_path, rgb_name = l
        name = rgb_name.rstrip("_rgb.jpg")
        pbar.set_description(f"[INFO] Processing {name}")

        rgb_path = os.path.join("tools", obj_dir_path, rgb_name)
        depth_path =  os.path.join("tools", obj_dir_path, name + '_depth.png')
        mask_path =  os.path.join(img_dir_path, obj_dir_path, f"{name}_label.mat") 
        affordances_path =  os.path.join(img_dir_path, obj_dir_path, f"{name}_label_rank.mat") 
        
        # load the object mask
        mask_obj = loadmat(mask_path)
        affordances_ids = np.unique(mask_obj['gt_label']) # all affordances present
        affordances_ids = np.delete(affordances_ids, 0) # remove 0 (indicates background)
        mask = (mask_obj['gt_label'] > 0).astype(np.uint8) # (480, 640)

        # compute the area, bbox and contours from the binary mask
        fortran_gt_binary_mask = np.asfortranarray(mask)
        encoded_ground_truth = pycoco_mask.encode(fortran_gt_binary_mask)
        area = pycoco_mask.area(encoded_ground_truth)
        bbox = pycoco_mask.toBbox(encoded_ground_truth).tolist()
        contours = measure.find_contours(mask, 0.5)
        category_id = [c['id'] for c in annot['categories'] if c['name'] == obj_dir_path][0]
        
        annotation = {
            'id': i + 1, # single-object scenes so same as image_id
            'image_id': i + 1,
            'category_id': category_id,
            'bbox': bbox, # [x,y,w,h]
            'area': area,
            'segmentation': [],
            'affordances': [],
            'iscrowd': 0
        }

        # add contours in Polygon format
        for contour in contours:
            contour = np.round(np.flip(contour, axis=1), 2)
            segmentation = contour.ravel().tolist()
            annotation['segmentation'].append(segmentation)

        # add affordances in Polygon format
        aff_obj = loadmat(affordances_path)
        for aff_id in affordances_ids:
            aff_mask = (aff_obj['gt_label'][:, :, aff_id-1] == 1).astype(np.uint8)
            contours = measure.find_contours(aff_mask, 0.5)
            polygon_mask = []
            for contour in contours:
                contour = np.round(np.flip(contour, axis=1), 2)
                segmentation = contour.ravel().tolist()
                polygon_mask.append(segmentation)
            aff = {
                'aff_id': aff_id, # affordance id
                'segmentation': polygon_mask # mask
            }
            annotation['affordances'].append(aff)

        annot['annotations'].append(annotation) # end of annotation

        # add image instance
        image = {
            'id': i + 1,
            'file_name': rgb_path,
            'depth_name': depth_path,
            'width': 640,
            'height': 480
        }
        annot['images'].append(image)
        
    if save_path is None:
        save_path = dataset_path
    save_path = os.path.join(save_path, f"UMD_{stype}_{split}_annotations.json")
    print(f'[INFO] Saving "{split}" annotation file at "{save_path}".' ) 
    with open(save_path, "w") as outfile:
        json.dump(annot, outfile, cls=NpEncoder)


#############################################################################
# --------------------------- CORNELL DATASET ----------------------------- #
#############################################################################

def generate_COCO_from_Cornell(dataset_path, save_path=None):
    """ Produces a COCO formatted JSON file from the Cornell grasp dataset and saves it in the dataset's 
    root directory for both the training and testing split.

    @param dataset_path: (str) path to Cornell grasp dataset root directory.
    @param save_path: (str, Optional) path where to save JSON file, if left at None the file will be saved
                      in the 'dataset_path' folder.

    TODO consider assigning object categories to images.
    """
    print(f'[INFO] Creating annotations for the Cornell grasp dataset (dataset_path: "{dataset_path}") ...') 

    # ---------------- set images and annotations ---------------
    total_files = [(subdir.lstrip(dataset_path), f) for subdir, dirs, files in os.walk(dataset_path) for f in files if f.endswith("r.png") and not subdir.endswith("backgrounds")]
    train_files, test_files = train_test_split(total_files, test_size=0.1, random_state=42)

    get_cornell_annotation(dataset_path, train_files, split="training", save_path=save_path)
    get_cornell_annotation(dataset_path, test_files, split="testing", save_path=save_path)

def get_cornell_annotation(dataset_path, files, split, save_path):
    annot = {'info':{}, 'licenses':[], 'images':[], 'categories':[], 'annotations':[]} 

    # -----------------  set info parameters -----------------
    annot['info']['decription'] = "Cornell Grasping Dataset"
    annot['info']['url'] = "https://www.kaggle.com/datasets/oneoneliu/cornell-grasp"
    annot['info']['version'] = "1.0"
    annot['info']['year'] = 2019

    grasp_counter = 0  # keeps track of the number of grasps for each object instance
    pbar = tqdm(files, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')
    for i, l in enumerate(pbar): # loops through each file path
        path = l[0] # the folder path
        name = l[1].rstrip("r.png")# the file name
        pbar.set_description(f"[INFO] Processing {name}")

        rgb_path = os.path.join(path, f"{name}r.png")  # path to RGB image from dataset root directory
        
        # get grasps
        grasp_path = os.path.join(dataset_path, path, f"{name}cpos.txt")
        grasps = parse_grasps(grasp_path, invert=True)
        grasps_id = [grasp_counter+g+1 for g in range(len(grasps))]
        grasp_counter += len(grasps)

        annotation = {
            'id': i + 1,
            'image_id': i + 1,
            'grasps': grasps, # [cx,cy,w,h,t]
            'grasps_id': grasps_id,
        }
        annot['annotations'].append(annotation) # end of annotation

        # add image instance
        image = {
            'id': i + 1,
            'file_name': rgb_path,
            'width': 640,
            'height': 480
        }
        annot['images'].append(image)
    
    if save_path is None:
        save_path = dataset_path
    save_path = os.path.join(save_path, f"Cornell_{split}_annotations.json")
    print(f'[INFO] Saving "{split}" annotation file at "{save_path}".' ) 
    with open(save_path, "w") as outfile:
        json.dump(annot, outfile, cls=NpEncoder)


#############################################################################
# --------------------------- OCID GRASP DATASET -------------------------- #
#############################################################################

def generate_COCO_from_OCID(dataset_path, split="training", save_path=None):
    """ Produces a COCO formatted JSON file from the OCID grasp dataset and saves it in the dataset's 
    root directory.

    @param dataset_path: (str) path to OCID grasp dataset root directory.
    @param split: (str) whether to generate annotations for the 'training' or 'validation' data splits provided.
    @param save_path: (str, Optional) path where to save JSON file, if left at None the file will be saved
                      in the 'dataset_path' folder.
    """
    print(f'[INFO] Creating "{split}" annotations for the OCID grasp dataset (dataset_path: "{dataset_path}") ...') 
    annot = {'info':{}, 'licenses':[], 'images':[], 'categories':[], 'annotations':[]} 

    # -----------------  set info parameters -----------------
    annot['info']['decription'] = "Object Clutter Indoor Dataset (OCID) with Grasps"
    annot['info']['url'] = "https://github.com/stefan-ainetter/grasp_det_seg_cnn"
    annot['info']['version'] = "1.0"
    annot['info']['year'] = 2019

    from data.OCID_grasp.OCID_class_dict import cnames

    # -------------------  set object categories ---------------------
    for name, i in cnames.items():
        if i == '0':  # skip 'background' class
            continue
        category = {
            'id': int(i),
            'name': name,
            'supercategory': name
        }
        annot['categories'].append(category)

    # ---------------- set images and annotations ---------------
    # get the file paths for the chosen data split (val or train)
    with open(os.path.join(dataset_path, "data_split", f"{split}_0.txt")) as f:
        file_paths = f.readlines()

    instance_counter = 0  # keeps track of number of object instances in a scene
    grasp_counter = 0  # keeps track of the number of grasps for each object instance
    pbar = tqdm(file_paths, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')
    for i, l in enumerate(pbar): # loops through each file path
        path = l.split(',')[0] # the folder path
        name = l.split(',')[1].strip() # the RGB file name
        pbar.set_description(f"[INFO] Processing {name}")

        rgb_path = os.path.join(path, "rgb", name)  # path to RGB image from dataset root directory
        depth_path = os.path.join(path, "depth", name)
        mask_path =  os.path.join(dataset_path, path, "seg_mask_instances_combi", name)  # complete path to instance mask
        class_path = os.path.join(dataset_path, path, "labels.txt") # complete path to image class labels

        img_mask = cv2.imread(os.path.join(mask_path), cv2.IMREAD_GRAYSCALE) # open instance mask
        colours = np.unique(img_mask.flatten())[1:].astype(np.uint8) # get all the unique objects (i.e. colours)

        # open and parse image class labels file into a list
        # note that classes of each object are given in order they are added to scene
        with open(class_path) as f:
            labels = f.readlines()
            labels = labels[0].strip().split(',')

        # for each object instance in the scene
        for pos in range(len(colours)):
            instance_counter += 1
            # get a binary mask of the object
            new_mask = img_mask.copy()
            new_mask[new_mask==colours[pos]] = 255
            new_mask[new_mask!=255] = 0
            new_mask = new_mask.astype(np.uint8)
            # compute the area, bbox and contours from the binary mask
            fortran_gt_binary_mask = np.asfortranarray(new_mask)
            encoded_ground_truth = pycoco_mask.encode(fortran_gt_binary_mask)
            area = pycoco_mask.area(encoded_ground_truth)
            bbox = pycoco_mask.toBbox(encoded_ground_truth).tolist()
            contours = measure.find_contours(new_mask, 0.5)
            category_id = int(labels[colours[pos]-1])
            
            # get grasps
            grasp_path = os.path.join(dataset_path, path, "Annotations_per_class", os.path.splitext(name)[0], str(category_id), os.path.splitext(name)[0] + '.txt')
            grasps = parse_grasps(grasp_path)
            # now filter by instance (i.e check if grasp is this particular object not just class by checking if it is within the bbox)
            x1, x2, y1, y2 = bbox[0], bbox[0]+bbox[2], bbox[1], bbox[1]+bbox[3]
            if len(grasps) > 0:
                grasps = [g for g in grasps if (x1 <= g[0] <= x2 and y1 <= g[1] <= y2)]
                grasps_id = [grasp_counter+g+1 for g in range(len(grasps))]
                grasp_counter += len(grasps)
            else:
                grasps_id = []

            annotation = {
                'id': instance_counter,
                'image_id': i + 1,
                'category_id': category_id,
                'bbox': bbox, # [x,y,w,h]
                'area': area,
                'grasps': grasps, # [cx,cy,w,h,t]
                'grasps_id': grasps_id,
                'segmentation': [],
                'iscrowd': 0
            }
            # add contours in Polygon format
            for contour in contours:
                contour = np.round(np.flip(contour, axis=1), 2)
                segmentation = contour.ravel().tolist()
                annotation["segmentation"].append(segmentation)
            annot['annotations'].append(annotation)

        # add image instance
        image = {
            'id': i + 1,
            'file_name': rgb_path,
            'depth_name': depth_path,
            'width': 640,
            'height': 480
        }
        annot['images'].append(image)
    
    if save_path is None:
        save_path = dataset_path
    save_path = os.path.join(save_path, f"OCID_{split}_annotations.json")
    print(f'[INFO] Saving "{split}" annotation file at "{save_path}".' ) 
    with open(save_path, "w") as outfile:
        json.dump(annot, outfile, cls=NpEncoder)


#############################################################################
# --------------------------- UTILITY FUNCTIONS -------------------------- #
#############################################################################

def parse_grasps(grasp_path, invert=False):
    """ Reads a grasp annotation file in Cornell dataset-type format, where each grasp is annoted in four lines representing
    four vertices in 'x, y' format. Each grasp is parsed and converted into a list of grasps of (cx, cy, w, h, t) format. 

    @param grasp_path: (str) path to the grasp annotation file to be open and parsed.
    @return: (list, tuple) a list of grasps in (cx, cy, w, h, t) format, where theta is in the range [-pi/2, pi/2].
    """
    grasps = [] # to store final grasps
    with open(grasp_path) as f:
        grasp_annots = f.readlines()
        grasp_rect = []  # to store the vertices of a single grasp rectangle
        for i, l in enumerate(grasp_annots):
            # parse the (x,y) co-ordinates of each grasp box vertice
            xy = l.strip().split()
            grasp_rect.append((float(xy[0]), float(xy[1])))
            if (i + 1) % 4 == 0: # create a grasp every 4 vertices
                if not np.isnan(grasp_rect).any():
                    cx, cy, w, h, theta = vertices_to_5D_grasp_pose(grasp_rect, invert=invert)
                    grasps.append((cx, cy, w, h, theta))
                    grasp_rect = []  # reset current grasp rectangle after 4 vertices have been read
    return grasps

def vertices_to_5D_grasp_pose(bbox, invert=False):
    """ Given four (x,y) vertices of a grasp rectangle, returns the (cx, cy, w, h, t) parameters of the grasp pose.
    Note that references include https://www.sciencedirect.com/science/article/pii/S0921889021000427 and
    https://github.com/skumra/robotic-grasping/blob/master/utils/dataset_processing/grasp.py.
    
    @param bbox: (list, tuple) a grasp rectangle as a list of four vertices where each vertex is in (x, y) format.
    @invert: (bool) invert the width and height (Cornell and OCID annotate slightly differently) set True for Cornell.
    @return: (tuple) a tuple (x, y, w, h, t) denoting the 'x, y' centre point, 'w' width, 'h' height and
                't' rotation of the given grasp rectangle. Note that theta is calculated to be in the range [-pi/2, pi/2].
    """
    x1, x2, x3, x4 = bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0]
    y1, y2, y3, y4 = bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1]
    cx, cy = (x1 + x2 + x3 + x4) / 4, (y1 + y2 + y3 + y4) / 4
    h = np.sqrt(np.power((x2 - x1), 2) + np.power((y2 - y1), 2))
    w = np.sqrt(np.power((x3 - x2), 2) + np.power((y3 - y2), 2))
    if invert:
        theta = (np.arctan2((y2 - y1), (x2 - x1)) + np.pi / 2) % np.pi - np.pi / 2
        return round(cx, 3), round(cy, 3), round(h, 3), round(w, 3), round(theta, 5)
    theta = (np.arctan2((y2 - y1), (x2 - x1))) % np.pi - np.pi / 2  # calculate theta [-pi/2, pi/2]
    return round(cx, 3), round(cy, 3), round(w, 3), round(h, 3), round(theta, 5)

class NpEncoder(json.JSONEncoder):
    """ Transforms JSON data from dictionaries into acceptable datatypes if applicable. """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def read_lines_from_text_file(file_path):
    """ Reads each line from a text file and store them in a list. """
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()  # removes '\n'
    return lines

def collate_fn(batch):
    """ Function for PyTorch data loader to load images with batch size that is > 1. """
    return tuple(zip(*batch))


#############################################################################
# --------------------------- TRANSFORMATIONS ----------------------------- #
#############################################################################

class Compose: 
    def __init__(self, transforms):
        """ Custom Compose function for a tuple (img, target) instead of just an img. """
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ImgCustomCrop(object):
    """ Custom crop function for an img. """
    def __init__(self, crop_size=[120,30,520,430]) -> None:
        self.crop_size = crop_size # [x_min, y_min, x_max, y_max]

    def __call__(self, img):
        new_img = img.crop((self.crop_size[0], self.crop_size[1], self.crop_size[2], self.crop_size[3]))
        return new_img

class MaskCustomCrop(object):
    """ Custom crop function for a mask. """
    def __init__(self, top, left, output_size) -> None:
        self.top = top
        self.left = left
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, img):
        # calculate the remaining amount to crop from the right and bottom sides of images
        right = self.left + self.output_size[0]
        bottom = self.top + self.output_size[1]
        new_img = img[self.top:bottom, self.left:right]
        return new_img

class ToTensor(object):
    """ Converts the image, bounding boxes and masks into PyTorch Tensors. """
    def __call__(self, img, target):
        new_img = torchvision.transforms.ToTensor()(img).float()
        if 'boxes' in target:   
            target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        if 'masks' in target:
            target['masks'] = torch.as_tensor(target['masks'], dtype=torch.uint8)
        return new_img, target

class CustomCrop(object):
    """ Crop the image by a specified amount of pixels from the edges.
    @param output_size: (tuple or int) desired output size. If given an int - a square crop is made.
    @param top: (float) number of pixels to crop from the top.
    @param left: (float) number of pixels to crop from the left.
    """
    def __init__(self, top, left, output_size):
        self.top = top
        self.left = left
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, img, target):
        # calculate the remaining amount to crop from the right and bottom sides of images
        right = self.left + self.output_size[0]
        bottom = self.top + self.output_size[1]

        # adjust the grasp or bbox co-ordinates
        if 'grasps' in target: # if grasps or bboxes
            target['grasps'][:, 0] -= self.left  # change cx
            target['grasps'][:, 1] -= self.top  # change cy
        else:
            target['boxes'][:, 0] -= self.left  # change xmin
            target['boxes'][:, 2] -= self.left  # change xmax
            target['boxes'][:, 1] -= self.top  # change ymin
            target['boxes'][:, 3] -= self.top  # change ymax
            # crop mask
            target['masks'] = target['masks'][:, self.top:bottom, self.left:right]
            if 'affs' in target:
                target['affs'] = target['affs'][:, :, self.top:bottom, self.left:right]

        # crop the image
        img = img.crop((self.left, self.top, right, bottom))
        return img, target

class RandomVerticalFlip(object):
    """ Randomly applies a vertical flip on the image with a specified probability (p).
    @param p: (float) the probability with which the image is flipped
    """
    def __init__(self, p=0.25):
        self.p = p

    def __call__(self, img, target):
        img_center = np.array((img.size[1], img.size[0]))[::-1] / 2
        img_center = np.hstack((img_center, img_center))

        if random.random() < self.p:
            img = img.transpose(Transpose.FLIP_TOP_BOTTOM)  # flip image
            
            if 'grasps' in target: # if grasps or bboxes
                target['grasps'][:, 1] += 2 * (img_center[1] - target['grasps'][:, 1])  # flip cy
                target['grasps'][:, 4] *= -1  # invert theta
            else:
                target['boxes'][:, [1, 3]] += 2 * (img_center[[1, 3]] - target['boxes'][:, [1, 3]]) # flip y co-ords
                box_h = abs(target['boxes'][:, 1] - target['boxes'][:, 3])
                target['boxes'][:, 1] -= box_h  # get new tl y co-ords (ymin)
                target['boxes'][:, 3] += box_h  # get new br x co-ords (ymax)
                # transform mask
                target['masks'] =  target['masks'][:,::-1,:]
                if 'affs' in target:
                    target['affs'] =  target['affs'][:,:,::-1,:]

        return img, target

class RandomHorizontalFlip(object):
    """ Randomly applies a horizontal flip on the image with a specified probability (p).
    @param p: (float) the probability with which the image is flipped
    """
    def __init__(self, p=0.25):
        self.p = p

    def __call__(self, img, target):
        img_center = np.array((img.size[1], img.size[0]))[::-1] / 2
        img_center = np.hstack((img_center, img_center))

        if random.random() < self.p:
            img = img.transpose(Transpose.FLIP_LEFT_RIGHT)  # flip image

            if 'grasps' in target: # if grasps or bboxes
                target['grasps'][:, 0] += 2 * (img_center[0] - target['grasps'][:, 0])  # flip cx
                target['grasps'][:, 4] *= -1  # invert theta
            else:
                # transform bbox co-ordinates 
                target['boxes'][:, [0, 2]] += 2 * (img_center[[0, 2]] - target['boxes'][:, [0, 2]]) # flip x co-ords (note that tl x becomes tr x and br x becomes bl x)
                box_w = abs(target['boxes'][:, 0] - target['boxes'][:, 2])
                target['boxes'][:, 0] -= box_w  # get new tl x co-ords (xmin)
                target['boxes'][:, 2] += box_w  # get new br x co-ords (xmax)
                # transform mask
                target['masks'] =  target['masks'][:,:,::-1].copy()
                if 'affs' in target:
                    target['affs'] =  target['affs'][:,:,:,::-1].copy()

        return img, target

class RandomShift(object):
    """ Applies a random horizontal or vertical shift to the image.
      :param px: (float) the max amount of pixels the image can shift.
      :param shift: (str) specify whether to shift horizontally (x), vertically (y) or both.
    """
    def __init__(self, px=50, shift='both'):
        self.px = px
        self.shift = shift

    def __call__(self, img, target):
        y_shift, x_shift = 0, 0
        # choose a random amount of pixels to translate in chosen directions
        if self.shift == 'x' or self.shift == 'both':
            x_shift = random.randint(-self.px, self.px + 1)
        if self.shift == 'y' or self.shift == 'both':
            y_shift = random.randint(-self.px, self.px + 1)
        # translate image by num. of pixels
        translated_image = img.transform(img.size, Transform.AFFINE, (1, 0, x_shift, 0, 1, y_shift))

        if 'grasps' in target: # if grasps or bboxes
            # translate grasps by num. of pixels
            target['grasps'][:,:4] -= [x_shift, y_shift, 0, 0]
        else:
            # translate bboxes by num. of pixels
            target['boxes'][:,:4] -= [x_shift, y_shift, x_shift, y_shift]
            # translate mask
            rows, cols = target['masks'][0].shape
            M = np.float32([[1, 0, -x_shift],[0, 1, -y_shift]]) # transformation matrix
            target['masks'] = np.asarray([self.translate_mask(m, M, (cols, rows)) for m in target['masks']])
            if 'affs' in target:
                target['affs'] = np.asarray([np.asarray([self.translate_mask(a, M, (cols, rows)) for a in m]) for m in target['affs']])
        return translated_image, target
    
    def translate_mask(self, mask, M, shape):
        # M = transformation matrix
        dst = cv2.warpAffine(mask.astype(np.float32), M, shape)
        return dst.astype(np.bool)


if __name__ == '__main__':
    data_choice = '' # put 'ocid', 'cornell', 'umd'

    # NOTE - important to double-check paths depending on directory you are calling from
    if data_choice == 'ocid':
        # OCID dataset
        dataset_path = "../data/OCID_grasp"
        generate_COCO_from_OCID(dataset_path, "training", save_path="../data/")  # "training" or "validation"
    elif data_choice == 'umd':
        # UMD dataset
        dataset_path =  "../data/part-affordance-dataset"
        generate_COCO_from_UMD(dataset_path, split="training", stype="category", save_path="../data/") # category or novel
    elif data_choice == 'cornell':
        # Cornell dataset
        dataset_path = "../data/Cornell_grasp"
        generate_COCO_from_Cornell(dataset_path, save_path="../data/")
