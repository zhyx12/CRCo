import os
import torch


def get_visible_card_num():
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cards = os.environ.get('CUDA_VISIBLE_DEVICES')
        cards_num = cards.count(',') + 1
        assert cards_num <= torch.cuda.device_count(), 'CUDA_VISIBLE_DEVICES should be lower than real number'
        return cards_num
    else:
        return torch.cuda.device_count()


if __name__ == "__main__":
    print(get_visible_card_num())
