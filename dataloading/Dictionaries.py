import pickle
import transformer.Constants as c


class Dictionaries:
    def __init__(self, dict_file):
        with open(dict_file, "rb") as file:
            terminal_counter = pickle.load(file)
            path_counter = pickle.load(file)
            target_counter = pickle.load(file)

        self.dict_terminal = dict()
        self.dict_path = dict()
        self.dict_target = dict()

        # Adding special words to dicts
        for word, idx in [(c.PAD_WORD, c.PAD), (c.UNK_WORD, c.UNK)]:
            self.dict_terminal[word] = idx
            self.dict_path[word] = idx
            self.dict_target[word] = idx

        for word, idx in [(c.EOS_WORD, c.EOS), (c.BOS_WORD, c.BOS)]:
            self.dict_target[word] = idx

        # Adding dataset dict
        start_terminal = len(self.dict_terminal)
        start_path = len(self.dict_path)
        start_target = len(self.dict_target)

        self.dict_terminal.update({w: i + start_terminal for i, w in enumerate(sorted([w for w, c in terminal_counter.items()]))})
        self.dict_path.update({w: i + start_path for i, w in enumerate(sorted([w for w, c in path_counter.items()]))})
        self.dict_target.update({w: i + start_target for i, w in enumerate(sorted([w for w, c in target_counter.items()]))})

    def terminal2idx(self, terminal: str) -> int:
        if terminal in self.dict_terminal:
            return self.dict_terminal[terminal]
        return self.dict_terminal[c.UNK_WORD]

    def path2idx(self, path: str) -> int:
        if path in self.dict_path:
            return self.dict_path[path]
        return self.dict_path[c.UNK_WORD]

    def target2idx(self, target: str) -> int:
        if target in self.dict_target:
            return self.dict_target[target]
        return self.dict_target[c.UNK_WORD]

    @property
    def terminal_vocab_size(self) -> int:
        return len(self.dict_terminal)

    @property
    def path_vocab_size(self) -> int:
        return len(self.dict_path)

    @property
    def target_vocab_size(self) -> int:
        return len(self.dict_target)
