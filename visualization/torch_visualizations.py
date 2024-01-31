from functools import partial
from typing import Optional

import pytorch_grad_cam
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import numpy as np



def get_visualization_tool(visualization_type:str, model:torch.nn.Module,
                           target_model_layer:torch.nn.Module)->\
        pytorch_grad_cam.base_cam.BaseCAM:

    visualizators = {
        "GradCam": GradCAM,
        "GradCam++": GradCAMPlusPlus,
        "ScoreCam": ScoreCAM,
        "AblationCam": AblationCAM,
        "XGradCam": XGradCAM,
        "EigenCam": EigenCAM,
        "FullGrad": FullGrad
    }
    if visualization_type not in visualizators.keys():
        raise ValueError(f"Unknown visualization type: {visualization_type}")

    cam = visualizators[visualization_type](model=model, target_layers=[target_model_layer])
    return cam



def perform_visualization(visualizator:pytorch_grad_cam.base_cam.BaseCAM, input:torch.Tensor, input_rgb:np.ndarray,
                          target:Optional[int]=None)->np.ndarray:

    # check all parameters
    if len(input.shape) != 4:
        raise ValueError(f"Input should have 4 dimensions (with batch size), but has {len(input.shape)}")
    if target != None:
        target = [ClassifierOutputTarget(target)]
    # perform visualization
    cam_ = visualizator(input_tensor=input, targets=target)
    visualization = show_cam_on_image(input_rgb / 255., cam_.transpose(1, 2, 0), use_rgb=True)
    return visualization









if __name__ == "__main__":
    # example of the usage of visualization tools
    from pytorch_utils.models.CNN_models import Modified_EfficientNet_B1
    from feature_extraction.pytorch_based.face_recognition_utils import load_and_prepare_detector_retinaFace_mobileNet, \
        recognize_one_face
    from pytorch_utils.models.input_preprocessing import resize_image_saving_aspect_ratio, \
        EfficientNet_image_preprocessor
    from torchinfo import summary

    from PIL import Image
    EMO_CATEGORIES: dict = {
        "N": 0,
        "H": 1,
        "Sa": 2,
        "Su": 3,
        "F": 4,
        "D": 5,
        "A": 6,
        "C": 7,

        0: "N",  # neutral
        1: "H",  # happy
        2: "Sa",  # sad
        3: "Su",  # surprise
        4: "F",  # fear
        5: "D",  # disgust
        6: "A",  # angry
        7: "C",  # contempt
    }
    # params
    path_to_weights = "/work/home/dsu/tmp/radiant_fog_160.pth"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    face_detector =load_and_prepare_detector_retinaFace_mobileNet(device="cuda:0")
    preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=240),
                                            EfficientNet_image_preprocessor()]
    # load model and prepare it
    model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=8,
                                             num_regression_neurons=2)
    model.load_state_dict(torch.load(path_to_weights))
    # cut off last layer
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.to(device)

    summary(model, (2, 3, 224, 224))
    # load data
    image_paths = ["/work/home/dsu/tmp/angry_1.jpeg",
                     "/work/home/dsu/tmp/sad_1.jpg",
                     "/work/home/dsu/tmp/happy_1.jpg"]
    input_tensors = [recognize_one_face(img, face_detector) for img in image_paths]
    # convert to torch.Tensor
    input_tensors = [torch.from_numpy(img).permute(2, 0, 1) for img in input_tensors]
    for preprocessing_function in preprocessing_functions:
        input_tensors = [preprocessing_function(img) for img in input_tensors]
    input_tensors = [img.to(device) for img in input_tensors]




    target_layers = [model[-6].features[-1]]
    targets = None
    cam = get_visualization_tool(visualization_type="GradCam", model=model, target_model_layer=target_layers[0],
                                 use_cuda=True)

    input = input_tensors[0].unsqueeze(0)
    # load images one more tipe using PIL
    inputs_rgb = [np.array(Image.open(img_path)) for img_path in image_paths]
    inputs_rgb = [recognize_one_face(img, face_detector) for img in image_paths]
    inputs_rgb = [resize_image_saving_aspect_ratio(img, expected_size=240) for img in inputs_rgb]
    # GRADCAM process
    for idx, (input, input_rgb) in enumerate(zip(input_tensors, inputs_rgb)):
        input = input.unsqueeze(0)
        visualization = perform_visualization(cam, input, input_rgb, target=targets)
        model_outputs = model(input)
        model_outputs = torch.nn.functional.softmax(model_outputs, dim=-1).squeeze().cpu().detach().numpy()
        predicted_class = EMO_CATEGORIES[np.argmax(model_outputs)]
        print(f'Image number {idx}, model output: {model_outputs}, predicted class: {predicted_class}')
        # show image
        import matplotlib.pyplot as plt
        plt.imshow(visualization)
        plt.show()






