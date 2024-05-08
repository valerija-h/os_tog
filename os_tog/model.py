import os
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, ResNet50_Weights
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN_ResNet50_FPN_Weights
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import yaml
from libraries.affcorrs.models.aff_corrs import AffCorrs_V1


#############################################################################
# --------------------------- OBJECT RECOGNITION -------------------------- #
#############################################################################

def build_object_recognition_model(cfg):
    # build resnet50 triplet-branch network
    feature_extractor = FeatureExtractor() 
    model = TripletModel(feature_extractor, guided_embeddings=cfg.GUIDED_EMBEDDINGS)
    model = nn.DataParallel(model) # make model run in parallel
    return model

class TripletModel(nn.Module):
    def __init__(self, feature_extractor, guided_embeddings=False):
        super(TripletModel, self).__init__()
        self.ge = guided_embeddings
        self.feature_extractor = feature_extractor
        if self.ge :
            self.skip = nn.Sequential(nn.Identity()) # just forwards input

    def forward(self, anchor, pos, neg):
        # shared weights if guided embeddings is false
        if self.ge: # don't need network if using features
            af = self.skip(anchor)
        else:
            af = self.feature_extractor(anchor) # anchor features
        pf = self.feature_extractor(pos) # positive features
        nf = self.feature_extractor(neg) # negative features
        return af, pf, nf

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # load resnet50 pre-trained on ImageNet
        self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        to_remove = -1 #   # remove layers (final classification layer) - number of layers to remove (-1 = 1, -2 = 2)
        self.model = nn.Sequential(*list(self.model.children())[:to_remove]) # remove final clasification layer
    
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2)
        return x
    
def train_object_recognition_model(dataset, dataloader, model, criterion, optimizer, epoch, device, cfg, neg_mining=None):
    total_loss, total_acc = 0, 0
    model.train() # switch to training mode
    
    prog_bar = tqdm(dataloader)
    for _, img_triplet in enumerate(prog_bar):
        prog_bar.set_description(desc=f"Training Epoch {epoch}")

        # get anchors, positives and negatives
        anchor_img, pos_img, neg_img = img_triplet['anchor_img'], img_triplet['pos_img'], img_triplet['neg_img']
        anchor_img, pos_img, neg_img = anchor_img.to(device), pos_img.to(device), neg_img.to(device)
        anchor_img, pos_img, neg_img = Variable(anchor_img), Variable(pos_img), Variable(neg_img)
        # get embeddings
        E1, E2, E3 = model(anchor_img, pos_img, neg_img) # first forward pass

        # make it select the nearest product image (multi-prod image)
        if cfg.MULTI_PROD:
            for i in range(E1.size(dim=0)):
                current_img_feat = E2[i] # get features of matching image (a.k.a positive image)
                current_img_label = img_triplet['pos_label'][i].item() # get object id of matching image
                closest_prod_feature = E1[i] # stores feature of closest product image (a.k.a anchor) - keeps track of closest anchor
                min_dist = 1000 # random large number - shouldn't go too high cause features are normalized
                for prod_img_feat in dataset.prod_img_features[current_img_label]:
                    with torch.no_grad(): # no need to compute gradients here to save memory
                        prod_img_feat = torch.from_numpy(prod_img_feat).to(device) # turn product img into tensor
                        feature_distance = F.pairwise_distance(current_img_feat, prod_img_feat, 2)
                        if feature_distance < min_dist: # if close product image is found
                            min_dist = feature_distance
                            closest_prod_feature = prod_img_feat
                E1[i] = closest_prod_feature # update anchor features to closest product image for loss calculations

        # if using a negative hard-mining strategy
        if neg_mining == "hard" or neg_mining == "semi-hard":
            E3 = online_hard_negative_mining(E1, E2, E3, img_triplet['pos_label'].to(device), img_triplet['neg_label'].to(device), neg_mining)

        # compute distance between anchor + positive embedding and anchor + negative embedding
        dist_E1_E2 = F.pairwise_distance(E1, E2, 2)
        dist_E1_E3 = F.pairwise_distance(E1, E3, 2)

        # 1 means dist_E1_E2 > dist_E1_E3
        target = torch.FloatTensor(dist_E1_E2.size()).fill_(-1)
        target = target.to(device)
        target = Variable(target)
        # triplet loss
        loss = criterion(dist_E1_E2, dist_E1_E3, target)
        acc = accuracy(dist_E1_E2, dist_E1_E3)
        total_loss += loss
        total_acc += acc

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prog_bar.set_postfix(loss=loss.item(), acc=acc.item()*100)
    print(f"Finished training epoch {epoch} - avg. train loss: {(total_loss/len(dataloader)):.5f} and avg. train acc: {(total_acc/len(dataloader)*100):.3f}")

def accuracy(dista, distb):
    """ Compute number of triplets where positive_dist > negative_dist """
    margin = 0
    pred = (distb - dista - margin).cpu().data
    return (pred > 0).sum()*1.0/dista.size()[0]

def test_object_recognition_model(dataset, dataloader, model, criterion, epoch, device, cfg):
    with torch.no_grad():
        model.eval() # switch to evaluation mode
        accuracies = [0, 0, 0, 0]
        acc_threshes = [0, 0.2, 0.5, 1]
        total_loss, total_acc = 0, 0

        prog_bar = tqdm(dataloader)
        for _, img_triplet in enumerate(prog_bar):
            prog_bar.set_description(desc=f"Validating Epoch {epoch}")
            # get batches of anchors, positives and negatives
            anchor_img, pos_img, neg_img = img_triplet['anchor_img'], img_triplet['pos_img'], img_triplet['neg_img']
            anchor_img, pos_img, neg_img = anchor_img.to(device), pos_img.to(device), neg_img.to(device)
            anchor_img, pos_img, neg_img = Variable(anchor_img), Variable(pos_img), Variable(neg_img)
            # get embeddings
            E1, E2, E3 = model(anchor_img, pos_img, neg_img)

            # make it select the nearest product image (multi-prod image)
            if cfg.MULTI_PROD:
                for i in range(E1.size(dim=0)): # loop through batch size
                    current_img_feat = E2[i] # get features of matching image (a.k.a positive image)
                    current_img_label = img_triplet['pos_label'][i].item() # get object id of matching image
                    closest_prod_feature = E1[i] # stores feature of closest product image (a.k.a anchor) - keeps track of closest anchor
                    min_dist = 1000 # random large number - shouldn't go too high cause features are normalized
                    for prod_img_feat in dataset.prod_img_features[current_img_label]:
                        with torch.no_grad(): # no need to compute gradients here to save memory
                            prod_img_feat = torch.from_numpy(prod_img_feat).to(device) # turn product img into tensor
                            feature_distance = F.pairwise_distance(current_img_feat, prod_img_feat, 2)
                            if feature_distance < min_dist: # if close product image is found
                                min_dist = feature_distance
                                closest_prod_feature = prod_img_feat
                    E1[i] = closest_prod_feature # update anchor features to closest product image for loss calculations

            # compute distance between anchor + positive embedding and anchor + negative embedding
            dist_E1_E2 = F.pairwise_distance(E1, E2, 2) # positive difference
            dist_E1_E3 = F.pairwise_distance(E1, E3, 2) # negative difference

            # 1 means dist_E1_E2 > dist_E1_E3
            # y = 1 means first input should be ranked higher, and y=-1 means second input should be ranked higher so we use -1
            # MarginRankingLoss(x1,x2,y)=max(0,−y∗(x1−x2)+margin)
            target = torch.FloatTensor(dist_E1_E2.size()).fill_(-1)
            target = target.to(device)
            target = Variable(target)
            # triplet loss
            loss = criterion(dist_E1_E2, dist_E1_E3, target)
            acc = accuracy(dist_E1_E2, dist_E1_E3)
            total_loss += loss
            total_acc += acc

            for i in range(len(accuracies)): # accuracy at different thresholds
                prediction = (dist_E1_E3 - dist_E1_E2 - cfg.MARGIN * acc_threshes[i]).cpu().data
                prediction = prediction.view(prediction.numel())
                prediction = (prediction > 0).float()
                batch_acc = prediction.sum() * 1.0 / prediction.numel()
                accuracies[i] += batch_acc
            
            prog_bar.set_postfix(val_loss=loss.item(), val_acc=acc.item()*100)
        
        print(f'Finished validating epoch {epoch} - avg. test loss: {(total_loss / len(dataloader)):.5f} and avg. test acc: {(total_acc/len(dataloader)*100):.3f}')
        for i in range(len(accuracies)):
            print(f'Test accuracy with diff = {acc_threshes[i] * 100}% of margin: {((accuracies[i] / len(dataloader)) * 100):.3f}')

def inference_object_recognition_model(model, img, prodImgClass, prodImgFeats):
    """ Returns predicted product image and class. In the following lines we calculate eucledian distance between 
    predicted img feature and all reference images. """
    with torch.no_grad():
        feature = model(img)
    feature = feature.cpu().detach().numpy()
    
    # 1repeat testImgFeat as len(prodImgsFeats) by first index (repmat(testImgFeat,size(prodImgsFeats,1),1))
    testImgFeatsRep = np.tile(feature, (len(prodImgFeats), 1))
    # find distance between testImgFeat and prodImgsFeats sqrt(sum((INPUT-prodImgsFeats).^2, 2))
    featDists = np.sqrt(np.sum(np.power((testImgFeatsRep - prodImgFeats), 2), 1))
    featDists = np.stack((prodImgClass, featDists), axis=-1) #  stack distances with prod labels each row is [label, featureDistance]
    sortedFeatDists = np.array(sorted(featDists, key=lambda x: x[1])) # sort array by shortest distance
    predNnDist = sortedFeatDists[0][1] # store the minimum distance
    predLabels = sortedFeatDists[:,0][0:5] # stored pred labels
    return predNnDist, predLabels

#######################################################################
########################  NEGATIVE MINING  ############################
#######################################################################

def get_distance_matrix(matrix_a, matrix_b):
    """ Takes as input two matrices of size (batch_size, embedding_size) and outputs a eucledian distance matrix of size 
    (batch_size, batch_size)  one-line formula is dist_ij = sqrt(xi.xi - 2xi.yj + yj.yj) 
    Reference: https://jaykmody.com/blog/distance-matrices-with-numpy/ 
    """
    # get the dot product of each row
    a_row = torch.sum(matrix_a**2, axis=1) # xi*xi size:(batch_size)
    b_row = torch.sum(matrix_b**2, axis=1) # yj*yj size:(batch_size)

    # matrix multiplication between matrix_a and matrix_b
    # note that (batch_size, embedding_size) x (embedding_size, batch_size) = (batch_size, batch_size)
    mult_matrix = torch.matmul(matrix_a, matrix_b.T) # xi*yj size:(batch_size, batch_size)

    a_row = torch.reshape(a_row, (-1, 1)) # reshape (batch_size) -> (batch_size, 1)
    distance_matrix = torch.sqrt(a_row - 2*mult_matrix + b_row) # compute formula (pairwise distance matrix)
    return distance_matrix

def filter_valid_negatives(al, nl):
    # produces a filter matrix shows if an element of al[0] doesn't have label as nl[0]
    is_equal = torch.eq(torch.unsqueeze(al, 1), torch.unsqueeze(nl, 0))
    filtered = torch.logical_not(is_equal).float()
    return filtered

def filter_valid_semi_hard(neg_dist, pos_dist):
    filtered = torch.gt(neg_dist, pos_dist).float() # valid negative example is greater than postive
    return filtered

def online_hard_negative_mining(ae, pe, ne, pl, nl, mode="hard"): # select "hard" or "semi-hard"
    # get a distance matrix between anchor embeddings and negative embeddings
    neg_pairwise_dist = get_distance_matrix(ae, ne)

    # for each anchor filter every valid negative (that have different labels from anchor)
    neg_mask = filter_valid_negatives(pl, nl) # 1 means it is a valid negative example

    if mode == "semi-hard":
        # also filter samples where pos distance < neg distance (neg has to be greater)
        pos_pairwise_dist = get_distance_matrix(ae, pe)
        semi_hard_mask = filter_valid_semi_hard(neg_pairwise_dist, pos_pairwise_dist)
        neg_mask = torch.logical_and(neg_mask, semi_hard_mask).float()

    # add the max value of each row to invalid negative
    # basically these two functions adds the maximum distance to each invalid negative example but keep the valid ones the same
    max_neg_dist, _ = torch.max(neg_pairwise_dist, axis=1, keepdim=True)
    anchor_negative_dist = neg_pairwise_dist + max_neg_dist * (1.0 - neg_mask)

    _, neg_idxs = torch.min(anchor_negative_dist, axis=1) # hard negatives

    negative_chosen_embeddings = ne[neg_idxs] # select the hard-est negative for each anchor!
    # TODO - consider returning new labels too? Also can return pe, ae if different (if you want to implement hard positives)

    return negative_chosen_embeddings


#############################################################################
# ------------------------ AFFORDANCE RECOGNITION ------------------------- #
#############################################################################

def build_affordance_recognition_model(model_path):
    with open(model_path) as f: # load arguments
        args = yaml.load(f, Loader=yaml.CLoader)
    args['low_res_saliency_maps'] = False
    args['load_size'] = 256

    model = AffCorrs_V1(args)
    return model

def inference_affordance_recognition_model(model, ref_image, ref_parts, tar_image):
        ref_parts = np.array(ref_parts, dtype=np.bool)
        affordances = [None for _ in ref_parts]
        with torch.no_grad():
            model.set_source(ref_image, ref_parts, affordances)
            model.generate_source_clusters()
            model.set_target(tar_image)
            model.generate_target_clusters()
            parts_out, aff_out = model.find_correspondences()
        return parts_out, aff_out


#############################################################################
# ---------------------------- GRASP DETECTION ---------------------------- #
#############################################################################

def build_grasp_detection_model(num_classes, anchor_sizes, aspect_ratios):
    """ This function creates an returns a grasp detection model. """
    # load a model pre-trained on COCO with custom anchors
    aspect_ratios = aspect_ratios * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(backbone_name='resnet50', weights=ResNet50_Weights.IMAGENET1K_V1)
    model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_grasp_detection_model(model, data_loader, optimizer, epoch, device, train_hist):
    """ This function trains the grasp detection model for one epoch. """
    model.train()  # set model to train mode
    prog_bar = tqdm(data_loader, total=len(data_loader))
    prog_bar.set_description(desc=f"Training Epoch {epoch}")

    train_hist[epoch] = {'loss': [], 'cls_loss': [], 'bbox_loss': [], 'rpn_cls_loss': [], 'rpn_bbox_loss': []}
    for _, data in enumerate(prog_bar):
        # get images and targets and send to device (i.e. GPU)
        images, targets = data
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # get losses
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        # zero out gradients
        optimizer.zero_grad()
        # back propagation and adjust learning weights  
        losses.backward()
        optimizer.step()

        # update loss for each batch
        prog_bar.set_postfix(loss=loss_value)

        # store losses for history and reporting results
        train_hist[epoch]['loss'].append(loss_value)
        train_hist[epoch]['cls_loss'].append(loss_dict['loss_classifier'].item())
        train_hist[epoch]['bbox_loss'].append(loss_dict['loss_box_reg'].item())
        train_hist[epoch]['rpn_cls_loss'].append(loss_dict['loss_objectness'].item())
        train_hist[epoch]['rpn_bbox_loss'].append(loss_dict['loss_rpn_box_reg'].item())
    return train_hist

def inference_grasp_detection_model(model, img, confidence=0.9):
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
    pred_class = [i for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = np.array([i for i in list(pred[0]['boxes'].detach().cpu().numpy())])
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class


#############################################################################
# ------------------------- INSTANCE SEGMENTATION ------------------------- #
#############################################################################
    
def build_instance_segmentation_model(num_classes=2):
    """ This function creates an returns an instance segmentation model. """
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                    hidden_layer,
                                                    num_classes)
    return model

def train_instance_segmentation_model(model, data_loader, optimizer, epoch, device, train_hist):
    """ This function trains the instance segmentation model for one epoch. """
    model.train()  # set model to train mode
    prog_bar = tqdm(data_loader, total=len(data_loader))
    prog_bar.set_description(desc=f"Training Epoch {epoch}")

    train_hist[epoch] = {'loss': [], 'cls_loss': [], 'bbox_loss': [], 'mask_loss':[], 'rpn_cls_loss': [], 'rpn_bbox_loss': []}
    for _, data in enumerate(prog_bar):
        # get images and targets and send to device (i.e. GPU)
        images, targets = data
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # get losses
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        # zero out gradients, back propagation and adjust learning weights  
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # update loss for each batch
        prog_bar.set_postfix(loss=loss_value)

        # store losses for history and reporting results
        train_hist[epoch]['loss'].append(loss_value)
        train_hist[epoch]['cls_loss'].append(loss_dict['loss_classifier'].item())
        train_hist[epoch]['bbox_loss'].append(loss_dict['loss_box_reg'].item())
        train_hist[epoch]['mask_loss'].append(loss_dict['loss_mask'].item())
        train_hist[epoch]['rpn_cls_loss'].append(loss_dict['loss_objectness'].item())
        train_hist[epoch]['rpn_bbox_loss'].append(loss_dict['loss_rpn_box_reg'].item())
    return train_hist

def inference_instance_segmentation_model(model, img, confidence=0.9):
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
    masks = (pred[0]['masks']>0.5).squeeze(0).detach().cpu().numpy()
    pred_class = [i for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = np.array([i for i in list(pred[0]['boxes'].detach().cpu().numpy())])
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class


#############################################################################
# -------------------------------- MISC ----------------------------------- #
#############################################################################

def save_model(model, directory, model_name):
    """ Save a model checkpoint. """
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = os.path.join(directory, model_name)
    print("[INFO] Saving checkpoint '{}'".format(path))
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """ Load a model checkpoint. """
    if os.path.isfile(path):
        print("[INFO] Loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)
        print("[INFO] Loaded checkpoint '{}'".format(path))
    else:
        print("[ERROR] No checkpoint found at '{}'".format(path))
    return model