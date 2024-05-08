import os
import json
import torch
import torchvision
import numpy as np
from PIL import Image

import os_tog.model as m
from os_tog.utils import visualize

class OS_TOG():
    def __init__(self, cfg):
        self.cfg = cfg
        self.annotations = self.load_database()
        self.class_values = self.generate_rotation_classes() # for grasp detection

        self.load_models()
        
    def load_models(self):
        instance_model = m.build_instance_segmentation_model()
        instance_model = m.load_model(instance_model, os.path.join(self.cfg.MODEL_DIR, self.cfg.SEGMENTATION_MODEL_NAME))
        instance_model.eval()
        instance_model.to(self.cfg.DEVICE)
        self.instance_model = instance_model

        object_model = m.build_object_recognition_model(self.cfg)
        object_model  = m.load_model(object_model, os.path.join(self.cfg.MODEL_DIR, self.cfg.RECOGNITION_MODEL_NAME))
        object_model = object_model.module.feature_extractor
        object_model.eval()
        object_model.to(self.cfg.DEVICE)
        self.object_model = object_model

        grasp_model = m.build_grasp_detection_model(self.cfg.ROTATION_CLASSES+1, self.cfg.ANCHOR_SIZES, self.cfg.ASPECT_RATIOS)
        grasp_model = m.load_model(grasp_model, os.path.join(self.cfg.MODEL_DIR,self.cfg.GRASP_MODEL_NAME))
        grasp_model.eval()
        grasp_model.to(self.cfg.DEVICE)
        self.grasp_model = grasp_model

        affordance_model = m.build_affordance_recognition_model(self.cfg.AFFORDANCE_MODEL_NAME)
        self.affordance_model = affordance_model

    def load_database(self):
        """ Load the physical experiment JSON annotation file. """
        try:
            f = open(self.cfg.DATABASE_FILE)
            annotations = json.load(f)
            return annotations
        except:
            print("[ERROR] Unable to load JSON annotation file.")
            return None
        
    def get_prediction(self, scene_img, target_object, target_task):
        """
        @param scene_img (float): RGB image
        @param target_object (str): e.g. hammer
        @param target_task (str): e.g. transport
        """
        _, pred_boxes, pred_mask, _ = self.get_instance_segmentation_predictions(scene_img, confidence=0.9)

        if len(pred_boxes) > 0:
            ref_info = self.get_reference_object(target_object)
            ref_img = self.load_reference_image(ref_info)
            obj_idx = self.get_object_recognition_predictions(ref_img, scene_img, (pred_mask, pred_boxes))

            ref_aff = self.get_reference_aff_mask(target_task, ref_info)
            if self.cfg.MULTI_REF_AFF: # align affordance to object through rotations
                ref_aff, ref_img = self.get_nearest_affordance(ref_aff, ref_img, scene_img, (pred_mask[obj_idx], pred_boxes[obj_idx]))
            pred_aff = self.get_affordance_recognition_predictions(ref_img, ref_aff, scene_img, (pred_mask[obj_idx], pred_boxes[obj_idx]))

            grasps = self.get_valid_grasps(scene_img, pred_aff)    
            return grasps[0] # return final grasp
        else:
            print("[ERROR] Failure in instance segmentation, no objects detected.")

    def get_instance_segmentation_predictions(self, scene_img, confidence=0.9):
        image = torchvision.transforms.ToTensor()(scene_img) # convert image to tensor and put on GPU
        image = image.to(self.cfg.DEVICE)
        with torch.no_grad():
            y_pred = self.instance_model([image])[0]
        pred_mask = ((y_pred['masks'] > 0.5).detach().cpu().numpy()).astype(np.int32)
        pred_score = list(y_pred['scores'].detach().cpu().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
        pred_class = [i for i in list(y_pred['labels'].cpu().numpy())]
        pred_boxes = np.array([i for i in list(y_pred['boxes'].detach().cpu().numpy())])
        pred_mask = pred_mask[:pred_t+1] # (N, 1, 480, 640)
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        pred_score = pred_score[:pred_t+1]

        if self.cfg.VISUALIZE:
            visualize(scene_img, boxes=pred_boxes, masks=pred_mask, title="Instance Segmentation Predictions", figsize=(5,5))

        return pred_score, pred_boxes, pred_mask, pred_class

    def get_reference_object(self, object_choice):
        """ Get an object annotation from the database. """
        object_infos = [ann for ann in self.annotations['objects'] if ann['name'] == object_choice]
        if len(object_infos) == 0:
            print("[ERROR] No such object in the database.")
            return None
        else: 
            object_info = object_infos[0]
            print(f"[INFO] Found object '{object_info['name']}' in database.")
            return object_info
    
    def load_reference_image(self, object_info, with_mask=True):
        """ Load an image of an object from the database. """
        ref_img = Image.open(os.path.join(self.cfg.DATABASE_DIR, object_info["img_path"]))
        if with_mask:
            mask = np.array(Image.open(os.path.join(self.cfg.DATABASE_DIR, object_info["mask_path"])))
            ref_img = apply_mask(ref_img, (mask[np.newaxis, :, :]/255.0))
        
        if self.cfg.VISUALIZE:
            visualize(np.array(ref_img), title="Reference Object", figsize=(5,5))
        return ref_img
    
    def get_object_recognition_predictions(self, ref_img, scene_img, segm_preds):
        ref_feature = self.get_reference_feature(ref_img) # [1, 2048]
        pred_masks, pred_boxes = segm_preds
        scene_obj_features, scene_objs_idxs = self.get_object_features(scene_img, pred_boxes, pred_masks)
        _ , obj_idx =  self.find_nearest_obj_idx(ref_feature, scene_obj_features, scene_objs_idxs) # get most similar object idx

        if self.cfg.VISUALIZE:
            visualize(scene_img, boxes=np.expand_dims(pred_boxes[obj_idx], axis=0), masks=pred_masks[obj_idx], title="Closest Object to Object Reference", figsize=(5,5))
        return obj_idx
    
    def get_reference_feature(self, ref_img):
        normalize_transform = torchvision.transforms.Normalize(self.cfg.NORM_MEAN, self.cfg.NORM_STD)
        tensor_transform  = torchvision.transforms.ToTensor()

        # preprocess reference image to get features
        ref_img_resized = torchvision.transforms.Resize((256, 256))(ref_img)
        ref_img_tensor = tensor_transform(ref_img_resized)
        ref_img_normalize = normalize_transform(ref_img_tensor)
        processed_ref_img = ref_img_normalize[None, :, :, :]
        processed_ref_img = processed_ref_img.to(self.cfg.DEVICE)
        
        # inference model
        with torch.no_grad():
            feature = self.object_model(processed_ref_img)
        ref_feature = feature.cpu().detach().numpy() # [1, 2048]
        return ref_feature
    
    def get_object_features(self, rgb_image, pred_boxes, pred_mask):
        normalize_transform = torchvision.transforms.Normalize(self.cfg.NORM_MEAN, self.cfg.NORM_STD)
        tensor_transform  = torchvision.transforms.ToTensor()

        predImgFeatures = np.zeros((len(pred_boxes), 2048)) # key is obj idx, value is 2048dim feature vector
        predImgIdxs = np.zeros((len(pred_boxes)))
        # ----- IMAGE MATCHING ----
        for pred_idx, bbox in enumerate(pred_boxes): # pre-process each prediction
            x1, y1, x2, y2 = bbox.astype(np.int32) 
            object_mask = apply_mask(rgb_image, pred_mask[pred_idx]) # apply mask on image
            object_crop = object_mask.crop((x1, y1, x2, y2)) # crop bounding box
            # preprocess observation image to get features
            object_padded = pad_PIL_image(object_crop) # pad to reach (256, 256, 3) image
            object_tensor = tensor_transform(object_padded)
            object_normalize = normalize_transform(object_tensor)
            processed_obj_img = object_normalize[None, :, :, :]
            processed_obj_img = processed_obj_img.to(self.cfg.DEVICE)
            with torch.no_grad():
                feature = self.object_model(processed_obj_img)
            predImgFeatures[pred_idx] = feature.cpu().detach().numpy()
            predImgIdxs[pred_idx] = pred_idx
        return predImgFeatures, predImgIdxs
    
    def find_nearest_obj_idx(self, ref_feature, predicted_features, predicted_idxs):
        """ Returns closest to reference object. """
        testImgFeatsRep = np.tile(ref_feature, (len(predicted_features), 1))
        featDists = np.sqrt(np.sum(np.power((testImgFeatsRep - predicted_features), 2), 1))
        featDists = np.stack((predicted_idxs, featDists), axis=-1)
        sortedFeatDists = np.array(sorted(featDists, key=lambda x: x[1]))
        predNnDists = sortedFeatDists[:,1][0]
        predIdxs = (sortedFeatDists[:,0][0]).astype(np.int32)
        return predNnDists, predIdxs
    
    def get_reference_aff_mask(self, task_choice, ref_info):
        task_infos = [t for t in self.annotations['tasks'] if t["name"] == task_choice]
        if len(task_infos) == 0:
            print("[ERROR] Unable to find task in database.")
            return None
        task_info = task_infos[0]

        if "NOT" in task_info['constraints']:
            # check if it has the region it doesn't want grasped
            to_remove = task_info['constraints'].split(" ")[1]
            aff_infos = [ann for ann in ref_info["affordances"] if ann["name"] == to_remove]
            if len(aff_infos) == 0: # grasp any where on the object if the no region isn't there
                mask = np.array(Image.open(os.path.join(self.cfg.DATABASE_DIR, ref_info["mask_path"]))) / 255.0
            else: # get 
                object_mask = np.array(Image.open(os.path.join(self.cfg.DATABASE_DIR, ref_info["mask_path"]))) / 255.0
                aff_info = aff_infos[0]
                remove_mask = np.array(Image.open(os.path.join(self.cfg.DATABASE_DIR, aff_info["mask_path"]))) / 255.0
                mask = np.logical_and(np.logical_not(remove_mask), object_mask)
        elif task_info['constraints'] == "": # no constraints
            mask = np.array(Image.open(os.path.join(self.cfg.DATABASE_DIR, ref_info["mask_path"]))) / 255.0 # object mask
        else:
            aff_infos = [ann for ann in ref_info["affordances"] if ann["name"] == task_info['constraints']]
            if len(aff_infos) == 0:
                print("[ERROR] Unable to find affordance for this task.")
                return None
            aff_info = aff_infos[0]

            mask = np.array(Image.open(os.path.join(self.cfg.DATABASE_DIR, aff_info["mask_path"]))) / 255.0
        return mask.astype(np.int32)
    
    def get_nearest_affordance(self, ref_aff, ref_img, obs_img, segm_preds):
        pred_masks, pred_boxes = segm_preds
        resize_transform = torchvision.transforms.Resize((256,256))
        # step 1 - preprocess reference affordance mask and
        ref_img_resized = resize_transform(ref_img) # (256, 256, 3)
        # step 2 - preprocess object and KEEP OFFSETS
        x1, y1, x2, y2 = pred_boxes.astype(np.int32) 
        object_mask = apply_mask(obs_img, pred_masks) # apply mask on image
        object_crop = object_mask.crop((x1, y1, x2, y2)) # crop bounding box
        object_padded, _ = pad_PIL_image(object_crop, return_offset=True, ds=300) # pad to reach (256, 256, 3) image
        object_padded = resize_transform(object_padded) # CHANGED
        
        closest_idx = 1
        closest_dist = 10000.0
        rotation = 45
        for i in range(1, int(360/rotation)):
            rotated_img = ref_img_resized.rotate(i*rotation) 
            dist = np.average(np.sqrt(np.sum(np.power((np.array(rotated_img) - np.array(object_padded)), 2), 1)))
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i
        ref_img = ref_img_resized.rotate(closest_idx*rotation)
        ref_aff_PIL = torchvision.transforms.ToPILImage()(ref_aff)
        ref_aff_PIL = resize_transform(ref_aff_PIL)
        ref_aff = np.array(ref_aff_PIL.rotate(closest_idx*rotation))

        return ref_aff, ref_img
    
    def get_affordance_recognition_predictions(self, ref_img, ref_aff, obs_img, segm_preds):
        pred_masks, pred_boxes = segm_preds
        resize_transform = torchvision.transforms.Resize((256,256))
        # step 1 - preprocess reference affordance mask and
        ref_aff_PIL = torchvision.transforms.ToPILImage()(ref_aff)
        ref_aff_resized = np.array(resize_transform(ref_aff_PIL)).astype(np.bool8) # (256, 256)
        ref_img_resized = resize_transform(ref_img) # (256, 256, 3)
        # step 2 - preprocess object and KEEP OFFSETS
        x1, y1, x2, y2 = pred_boxes.astype(np.int32) 
        object_mask = apply_mask(obs_img, pred_masks) # apply mask on image
        object_crop = object_mask.crop((x1, y1, x2, y2)) # crop bounding box
        object_padded, (pl, pt, pr, pb) = pad_PIL_image(object_crop, return_offset=True, ds=300) # pad to reach (256, 256, 3) image
        object_padded = resize_transform(object_padded) # CHANGED
        
        try: # get predictions from image
            # make sure mask has 1 axis at the start and is type bool
            aff_pred, _ = m.inference_affordance_recognition_model(self.affordance_model, ref_img_resized, [ref_aff_resized], object_padded)
        except Exception as e:
            print(f"[ERROR] Affordance recognition failed: {e}")
            return None
        
        pred_mask = aff_pred[0]
        unpad_mask = torchvision.transforms.ToPILImage()(pred_mask.astype(np.int32)) # PIL IMAGE
        unpad_mask = torchvision.transforms.Resize((300,300))(unpad_mask)
        pred_mask = np.array(unpad_mask)
        unpad_mask = (pred_mask[pt:(pred_mask.shape[1]-pb),pl:(pred_mask.shape[0]-pr)]).astype(np.int32)
        unpad_mask = torchvision.transforms.ToPILImage()(unpad_mask)
        uncrop_mask = torchvision.transforms.Pad((x1, y1, 640-x2, 480-y2))(unpad_mask) # left, top, right, bottom
        uncrop_mask = np.array(uncrop_mask) # resize to original image

        if self.cfg.VISUALIZE:
            visualize(np.array(ref_img), masks=np.asarray([ref_aff]), title="Reference Affordance", figsize=(5,5)) # may be rotate if u chose MULTI_REF_AFF=True in cfg
            visualize(obs_img, masks=np.asarray([uncrop_mask]), title="Affordance Prediction", figsize=(5,5))
        return uncrop_mask

    def get_valid_grasps(self, scene_image, aff_mask):
        tensor_img = torchvision.transforms.ToTensor()(scene_image)
        tensor_img = tensor_img.to(self.cfg.DEVICE)
        pred_boxes, pred_class = m.inference_grasp_detection_model(self.grasp_model, tensor_img, confidence=0.5)

        grasps = [self.VOC_to_5D_grasp_pose(pred_boxes[i], pred_class[i].item()) for i in range(len(pred_boxes))]

        if self.cfg.VISUALIZE:
            visualize(scene_image,  grasps=np.array(grasps), title=f"All Grasp Predictions", figsize=(5,5))

        grasps = [g for g in grasps if aff_mask[int(g[1]), int(g[0])] == 1] # x, y, w, h, t
        if grasps == 0:
            print("[ERROR] No grasps in predicted task region.")
            return None
        
        if self.cfg.VISUALIZE:
            visualize(scene_image, grasps=np.array(grasps), title=f"Task-oriented Grasp Predictions", figsize=(5,5))
            visualize(scene_image,  grasps=np.array([grasps[0]]), title=f"Most Confident Grasp Prediction", figsize=(5,5))
        return np.array(grasps, dtype=np.float32)
    
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
        rot_size = (rot_range / self.cfg.ROTATION_CLASSES) * (np.pi / 180)  # diff. in rotation between each idx (radians)
        for i in range(self.cfg.ROTATION_CLASSES):
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

#############################################################################
# ---------------------------- UTILITY FUNCTIONS -------------------------- #
#############################################################################

def apply_mask(image, mask):
    """
    @param img (np.array float): image with shape [480,640,3]
    @param mask (np.array float): mask with shape [1,480,640]
    """
    mask = np.repeat(mask[0][:,:,None], 3, axis=-1)
    np_img = image * mask
    new_img = Image.fromarray(np.uint8(np_img))
    return new_img

def pad_PIL_image(image, ds=256, return_offset= False):
    """
    @param ds (int): desired output size (square)
    """
    w, h = image.size
    l, r = (ds-w)//2,(ds-w)//2
    t, b = (ds-h)//2,(ds-h)//2

    if (ds-w) % 2 != 0:
        l, r = (ds-w)//2 + 1,(ds-w)//2
    if (ds-h) % 2 != 0:
        t, b = (ds-h)//2 + 1,(ds-h)//2

    object_padded = torchvision.transforms.Pad((l, t, r, b))(image)
    if return_offset:
        return object_padded, (l, t, r, b)
    return object_padded