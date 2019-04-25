from model import NeuralStyleTransfer
import click
import tqdm
import os
import torch
from PIL import Image
import torchvision.transforms as transforms


@click.command()
@click.option('--epochs', default=2000, help='Number of epochs')
@click.option('--alpha', default=1, help='Content weight')
@click.option('--beta', default=1e6, help='Style weight')
@click.argument('style-image')
@click.argument('content-image')
@click.argument('target-image')
def main(epochs, alpha, beta, style_image, content_image, target_image):
    model = NeuralStyleTransfer()
    style_tensor = load_image(os.path.expanduser(style_image))
    content_tensor = load_image(os.path.expanduser(content_image))
    target_tensor = load_image(os.path.expanduser(target_image))
    style_features = model.forward(style_tensor)
    content_features = model.forward(content_tensor)

    optimizer = torch.optim.Adam([target_tensor], lr=0.003)
    for epoch in tqdm.tqdm(range(epochs)):
        target_features = model.forward(target_tensor)

        content_loss = 0
        for layer in model.content_layers:
            content_loss += torch.mean((target_features[layer] - content_features[layer])**2)

        style_loss = 0
        for layer in model.style_layers:
            # TODO: Optimize gram_matrix call for style_features
            style_loss += torch.mean((gram_matrix(target_features[layer]) - gram_matrix(style_features[layer]))**2)

        total_loss = alpha * content_loss + beta * style_loss

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()

    transform = transforms.ToPILImage()
    image = transform(target_tensor.squeeze())
    image.save(os.path.expanduser(target_image))


def gram_matrix(tensor):
    matrix = tensor.view(tensor.size(1), -1)
    return torch.mm(matrix, matrix.t())


def load_image(path):
    transform = transforms.ToTensor()
    image = Image.open(path).convert('RGB')
    tensor = transform(image)
    tensor.unsqueeze_(0)
    tensor.requires_grad_(True)

    return tensor


if __name__ == '__main__':
    main()
