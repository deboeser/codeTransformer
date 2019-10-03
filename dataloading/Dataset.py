import h5py
import torch
from torch.utils.data import Dataset
from numpy.random import shuffle
import transformer.Constants as c


def collate_fn(data):
    return data


class C2SDataSet(Dataset):
    def __init__(self, filedata, dicts, device="cpu"):
        super(Dataset, self).__init__()
        self.f = filedata
        with h5py.File(self.f, "r") as f:
            self.size = max([int(x) for x in list(f.keys())])

        self.dicts = dicts
        self.device = device

        # TODO: Remove the hardcoded numbers
        self.max_context_length = 200  # args.context_length
        self.max_terminal_length = 8  # args.terminal_length
        self.max_path_length = 16  # args.path_length
        self.max_target_length = 8  # args.target_length

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        with h5py.File(self.f, "r") as f:
            line = f[str(index)]["line"][()]

        path_context_string = line.split(" ")
        path_contexts_raw = [s.split(",") for s in path_context_string[1:] if len(s) > 3]

        starts, paths, ends, target = [], [], [], []
        starts_pos, paths_pos, ends_pos = [], [], []

        # Only consider the first max_context_length paths
        path_contexts_raw_indices = [i for i in range(len(path_contexts_raw))]
        shuffle(path_contexts_raw_indices)

        for index in path_contexts_raw_indices[:self.max_context_length]:
            path_context_raw = path_contexts_raw[index]
            if len(path_context_raw) != 3:
                continue

            # Processing the start terminal, path, end terminal
            start = [self.dicts.terminal2idx(t) for t in path_context_raw[0].split("|")[:self.max_terminal_length]]
            path = [self.dicts.path2idx(p) for p in path_context_raw[1].split("|")[:self.max_path_length]]
            end = [self.dicts.terminal2idx(t) for t in path_context_raw[2].split("|")[:self.max_terminal_length]]

            # Position IDs
            start_pos = [i+1 for i in range(len(start))]
            path_pos = [i+1 for i in range(len(path))]
            end_pos = [i+1 for i in range(len(end))]

            # Padding to fixed sizes
            start += [c.PAD] * (self.max_terminal_length - len(start))
            path += [c.PAD] * (self.max_path_length - len(path))
            end += [c.PAD] * (self.max_terminal_length - len(end))
            start_pos += [0] * (self.max_terminal_length - len(start_pos))
            path_pos += [0] * (self.max_path_length - len(path_pos))
            end_pos += [0] * (self.max_terminal_length - len(end_pos))

            starts.append(start)
            paths.append(path)
            ends.append(end)
            starts_pos.append(start_pos)
            paths_pos.append(path_pos)
            ends_pos.append(end_pos)

        # Creating the target
        target.append(c.BOS)
        for tar_s in path_context_string[0].split("|")[:self.max_target_length - 2]:
            target.append(self.dicts.target2idx(tar_s))
        target.append(c.EOS)
        targets_pos = [i+1 for i in range(len(target))]

        # Padding the target
        target += [c.PAD] * (self.max_target_length - len(target))
        targets_pos += [0] * (self.max_target_length - len(targets_pos))

        # Padding context length and create mask
        pad_length = self.max_context_length - len(starts)
        starts += [[c.PAD for i in range(self.max_terminal_length)]] * pad_length
        paths += [[c.PAD for i in range(self.max_path_length)]] * pad_length
        ends += [[c.PAD for i in range(self.max_terminal_length)]] * pad_length
        starts_pos += [[0 for i in range(self.max_terminal_length)]] * pad_length
        paths_pos += [[0 for i in range(self.max_path_length)]] * pad_length
        ends_pos += [[0 for i in range(self.max_terminal_length)]] * pad_length

        return (torch.tensor(starts, dtype=torch.long).to(self.device),
                torch.tensor(paths, dtype=torch.long).to(self.device),
                torch.tensor(ends, dtype=torch.long).to(self.device),
                torch.tensor(starts_pos, dtype=torch.long).to(self.device),
                torch.tensor(paths_pos, dtype=torch.long).to(self.device),
                torch.tensor(ends_pos, dtype=torch.long).to(self.device),
                torch.tensor(target, dtype=torch.long).to(self.device),
                torch.tensor(targets_pos, dtype=torch.long).to(self.device),)
