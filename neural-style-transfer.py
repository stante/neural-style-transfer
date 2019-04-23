from model import NeuralStyleTransfer
import click


@click.option('--epochs', default=2000, help='Number of epochs')
def main(epochs):
    model = NeuralStyleTransfer()

    for epoch in range(epochs):
        pass


if __name__ == '__main__':
    main()
