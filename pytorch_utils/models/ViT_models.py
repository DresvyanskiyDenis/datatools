from torch import nn
from torchinfo import summary
from transformers import DeiTImageProcessor, DeiTForImageClassification
import torch
from PIL import Image
import requests

class ViT_Deit_model(nn.Module):
    def __init__(self, num_classes):
        super(ViT_Deit_model, self).__init__()
        self.model = DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224")
        self.model.classifier = nn.Identity()
        self.tranformer_output = nn.Linear(768, 256)
        self.classifier = nn.Linear(256, num_classes)
        self.arousal = nn.Linear(256, 1)
        self.valence = nn.Linear(256, 1)
        self.num_labels = num_classes

    def forward(self, x):
        x = self.model(x)
        x = x[0] # convert from special output format from Hugging Face
        x = self.tranformer_output(x)
        y1 = self.classifier(x)
        y2 = self.arousal(x)
        y2 = torch.tanh(y2)
        y3 = self.valence(x)
        y3 = torch.tanh(y3)
        return y1, y2, y3



if __name__ == "__main__":
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = DeiTImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")

    model = ViT_Deit_model(num_classes=8)
    summary(model, input_size=(1, 3, 224, 224), device='cpu')
    print("-------------------------")
    print(model.classifier)

    input = image_processor(images=image, return_tensors="pt")['pixel_values']

    outputs = model(input)

    # model predicts one of the 1000 ImageNet classes

    print(outputs)