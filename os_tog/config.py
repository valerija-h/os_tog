class Config:
    # DATASET PARAMETERS
    DATASET_PATH = '../data'
    OCID_PATH = f'{DATASET_PATH}/OCID_grasp' # path to OCID grasp dataset
    OCID_ANNOT_PATH = f'{DATASET_PATH}/annotations/' # path to OCID grasp annotation file
    UMD_PATH = f'{DATASET_PATH}/part-affordance-dataset/'
    UMD_ANNOT_PATH = f'{DATASET_PATH}/annotations/'
    CORNELL_PATH = f'{DATASET_PATH}/Cornell_grasp/'  # path to Cornell grasp dataset
    CORNELL_ANNOT_PATH = f'{DATASET_PATH}/annotations/'

    # MODEL PARAMETERS
    MODEL_DIR = '../models'

    # TRAINING PARAMETERS  
    NUM_WORKERS = 4
    TRAIN_BATCH_SIZE = 4
    VAL_BATCH_SIZE = 4
    TEST_BATCH_SIZE = 1
    EPOCHS = 10  

class InstanceSegmentationConfig(Config):
    # TRAINING PARAMETERS
    EPOCHS = 1
    LEARNING_RATE = 0.0001

    # MODEL PARAMETERS
    MODEL_DIR = "../models/instance_segmentation"
    MODEL_NAME = "maskrcnn_OCID_t1"
    
    # TRAINING PARAMETERS
    TRAIN_BATCH_SIZE = 4

    # INFERENCE PARAMETERS
    CONFIDENCE = 0.9

class GraspDetectionConfig(Config):
    # DATASET PARAMETERS
    ROTATION_CLASSES = 12

    # MODEL PARAMETERS
    ANCHOR_SIZES = ((32,), (64,), (128,), (256,), (512,),)
    ASPECT_RATIOS = ((0.25, 0.5, 1.0),)
    MODEL_DIR = "../models/grasp_detection"
    MODEL_NAME = "fasterrcnn_cornell_t1"

    # TRAINING PARAMETERS
    EPOCHS = 5
    LEARNING_RATE = 0.0001

    # INFERENCE PARAMETERS
    CONFIDENCE = 0.5

class AffordanceRecognitionConfig(Config):
    # MODEL PARAMETERS
    MODEL_PATH = "../libraries/affcorrs/config/default_config.yaml"
    
    # INFERENCE PARAMETERS
    CONFIDENCE = 0.5

    # TRAINING PARAMETERS
    TEST_BATCH_SIZE = 4

class ObjectRecognitionConfig(Config):
    # DATASET PARAMETERS
    NORM_MEAN = (0.485, 0.456, 0.406) 
    NORM_STD = (0.229, 0.224, 0.225)

    # MODEL PARAMETERS
    MODEL_DIR = "../models/object_recognition"
    MODEL_NAME = "triplet_UMD_masked_t1"
    GUIDED_EMBEDDINGS = True  # changes dataset loader + triplet model
    MULTI_PROD = True  # changes training loop
    RANDOM_PROD = False # if to select first or random product/ref images
    FIXED_BLOCKS = 1
    NEG_MINING = "semi-hard" # "semi-hard" or "hard" or none
    
    # TRAINING PARAMETERS
    EPOCHS = 3
    LEARNING_RATE = 0.001 # learning rate
    MOMENTUM = 0.99 # SGD momentum
    MARGIN = 0.2  # only needed if using 'triplet_loss' (default is 0.2)
    TRAIN_BATCH_SIZE = 20
    VAL_BATCH_SIZE = 20


class OSTOGConfig(Config): # TODO - verify physical experiment parameters
    DATABASE_FILE = "../data/annotations/OSTOG_physical_experiments.json"
    DATABASE_DIR = "../data/OSTOG_physical_experiments/"
    VISUALIZE = True
    MULTI_REF_AFF = True # multiple references for aff recognition

    # MODEL PARAMETERS
    MODEL_DIR = "../models/os_tog"
    GRASP_MODEL_NAME = "grasp_detection_model.pt"
    SEGMENTATION_MODEL_NAME = "instance_segmentation_model.pt"
    RECOGNITION_MODEL_NAME = "object_recognition_model.pt"
    AFFORDANCE_MODEL_NAME = "../libraries/affcorrs/config/default_config.yaml"
    GUIDED_EMBEDDINGS = False
    MULTI_PROD = False
    FIXED_BLOCKS = 1
    NORM_MEAN = (0.485, 0.456, 0.406) 
    NORM_STD = (0.229, 0.224, 0.225)
    ROTATION_CLASSES = 12
    ANCHOR_SIZES = ((32,), (64,), (128,), (256,), (512,),)
    ASPECT_RATIOS = ((0.25, 0.5, 1.0),)
    
    # INFERENCE PARAMETERS
    CONFIDENCE = 0.5
