from model import NeuralStyleTransfer
import click
import torch
import tqdm


@click.command()
@click.option('--epochs', default=2000, help='Number of epochs')
def main(epochs):
    model = NeuralStyleTransfer()

    for epoch in tqdm.tqdm(range(epochs)):
        features = model.forward(torch.randn(1, 3, 224, 224))


if __name__ == '__main__':
    main()
