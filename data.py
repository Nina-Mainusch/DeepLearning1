from torch.utils.data import Dataset
from dlc_practical_prologue import generate_pair_sets


class P1Dataset(Dataset):
    def __init__(self, N, type):
        if type == "train":
            self.data, self.target, self.classes, _, _, _ = generate_pair_sets(N)
        elif type == "test":
            _, _, _, self.data, self.target, self.classes = generate_pair_sets(N)
        else:
            raise Exception

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.target[index], self.classes[index]
