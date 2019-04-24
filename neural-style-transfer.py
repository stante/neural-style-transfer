from model import NeuralStyleTransfer
import click
import tqdm
import os
from PIL import Image
import torchvision.transforms as transforms


@click.command()
@click.option('--epochs', default=2000, help='Number of epochs')
@click.argument('style-image')
@click.argument('content-image')
def main(epochs, style_image, content_image):
    model = NeuralStyleTransfer()
    style_tensor = load_image(os.path.expanduser(style_image))

    for epoch in tqdm.tqdm(range(epochs)):
        features = model.forward(style_tensor)


def load_image(path):
    transform = transforms.ToTensor()
    image = Image.open(path).convert('RGB')
    tensor = transform(image)
    tensor.unsqueeze_(0)
    tensor.requires_grad_(True)

    return tensor


if __name__ == '__main__':
    main()
