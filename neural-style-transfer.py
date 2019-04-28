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
    device = 'cpu'
    if torch.cuda.is_available:
        print("Using cuda ", torch.cuda.current_device())
        device = 'cuda'

    model = NeuralStyleTransfer()
    model.to(device)
    style_tensor = load_image(os.path.expanduser(style_image))
    style_tensor = style_tensor.to(device)
    content_tensor = load_image(os.path.expanduser(content_image))
    content_tensor = content_tensor.to(device)
    target_tensor = load_image(os.path.expanduser(target_image))
    style_features = model.forward(style_tensor)
    content_features = model.forward(content_tensor)

    optimizer = torch.optim.Adam([target_tensor], lr=0.003)

    print(content_tensor.shape)
    print(style_tensor.shape)
    print(target_tensor.shape)

    for epoch in range(epochs):
        target_features = model.forward(target_tensor)

        content_loss = 0
        for layer in model.content_layers:
            content_loss += torch.mean((target_features[layer] - content_features[layer])**2)

        style_loss = 0
        for layer in model.style_layers:
            # TODO: Optimize gram_matrix call for style_features
            _, d, h, w = target_features[layer].shape
            layer_style_loss = model.style_weights[layer] * torch.mean((gram_matrix(target_features[layer]) - gram_matrix(style_features[layer]))**2)
            style_loss += layer_style_loss / (d * w * h)

        total_loss = alpha * content_loss + beta * style_loss

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % 1 == 0:
            print("")
            print("C: {}".format(content_loss.item()))
            print("S: {}".format(style_loss.item()))
            print("T: {}".format(total_loss.item()))

    import numpy as np
    target_numpy = target_tensor.squeeze().cpu().detach().numpy()
    target_numpy = target_numpy.transpose(1, 2, 0)

    image = target_tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)


    #print(target_tensor.shape)
    #print(target_numpy.shape)
    #transform = transforms.ToPILImage()
    #target_tensor = target_tensor.squeeze().cpu()
    # target_tensor = target_tensor.clip(0, 1)
    #image = transform(target_tensor)
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.show()

    image = image * 255
    image = Image.fromarray(np.asarray(image, dtype=np.uint8), mode='RGB')
    #image.save(os.path.expanduser(target_image))


def gram_matrix(tensor):
    tensor = tensor.squeeze()
    matrix = tensor.view(tensor.shape[0], -1)
    gram = torch.mm(matrix, matrix.transpose(0, 1))

    return gram


#def gram_matrix(tensor):
#    matrix = tensor.view(tensor.size(1), -1)
#    return torch.mm(matrix, matrix.t())


def load_image(path):
    transform = transforms.Compose( [transforms.Resize((400, 592)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406),
                                             (0.229, 0.224, 0.225))])

    image = Image.open(path).convert('RGB')
    tensor = transform(image).unsqueeze(0)
    # tensor.unsqueeze_(0)
    tensor = tensor.to('cuda').requires_grad_(True)

    return tensor


if __name__ == '__main__':
    main()
