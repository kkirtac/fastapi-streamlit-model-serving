import torch
from albumentations.pytorch.transforms import ToTensorV2
from modules.transforms import TransformationsBase

# we are using the pixel mean std of ImageNet training set
# because our trained model used these values for image normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def inference(model, image):

    transform = TransformationsBase(scale_size=224, norm_mean=IMAGENET_MEAN, norm_std=IMAGENET_STD)

    # scaling and normalization transforms
    image = transform.eval_transform(image=image)["image"]
    image = transform.norm_transform(image=image)["image"]
    image = ToTensorV2()(image=image)["image"]

    # unsqueeze the 3-D tensor to obtain 4-D because pytorch model requires 4-D (batch dimension)
    scores = torch.nn.functional.softmax(model(image.unsqueeze(0)), dim=1).flatten()

    return {
        'full_visibility': scores[0].item(),
        'partial_visibility': scores[1].item(),
        'no_visibility': scores[2].item(),
    }