import yaml
import sys

from train import train
from infer import infer

if __name__ == '__main__':
    train_data = yaml.load(open('data/train_data.yml'), Loader=yaml.FullLoader)
    train(train_data)
    if len(sys.argv) > 1 and sys.argv[1] == '-t':
        train_data = yaml.load(open('data/train_data.yml'), Loader=yaml.FullLoader)
        train(train_data)
    else:
        for range in range(10):
            print('Bot: ', infer(input('user: ')))
