"""Data loaders for leave-one-out sequential recommendation."""

import torch
import torch.utils.data as data_utils


class TrainDataset(data_utils.Dataset):
    def __init__(self, id2seq, max_len):
        self.id2seq = id2seq
        self.max_len = max_len

    def __len__(self):
        return len(self.id2seq)

    def __getitem__(self, index):
        seq = self.id2seq[index]
        labels = [seq[-1]]
        tokens = seq[:-1][-self.max_len:]
        tokens = [0] * (self.max_len - len(tokens)) + tokens
        return torch.LongTensor(tokens), torch.LongTensor(labels)


class Data_Train:
    def __init__(self, data_train, args):
        self.u2seq = data_train
        self.max_len = args.max_len
        self.batch_size = args.batch_size
        self.id_seq = {}
        self.split_onebyone()

    def split_onebyone(self):
        idx = 0
        for seq in self.u2seq.values():
            for start in range(len(seq) - 1):
                self.id_seq[idx] = seq[:start + 2]
                idx += 1

    def get_pytorch_dataloaders(self):
        dataset = TrainDataset(self.id_seq, self.max_len)
        return data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=torch.cuda.is_available())


class ValDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user][-self.max_len:]
        seq = [0] * (self.max_len - len(seq)) + seq
        answer = self.u2answer[user]
        return torch.LongTensor(seq), torch.LongTensor(answer)


class Data_Val:
    def __init__(self, data_train, data_val, args):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2answer = data_val
        self.max_len = args.max_len

    def get_pytorch_dataloaders(self):
        dataset = ValDataset(self.u2seq, self.u2answer, self.max_len)
        return data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=torch.cuda.is_available())


class TestDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2seq_add, u2answer, max_len):
        self.u2seq = u2seq
        self.u2seq_add = u2seq_add
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user] + self.u2seq_add[user]
        seq = seq[-self.max_len:]
        seq = [0] * (self.max_len - len(seq)) + seq
        answer = self.u2answer[user]
        return torch.LongTensor(seq), torch.LongTensor(answer)


class Data_Test:
    def __init__(self, data_train, data_val, data_test, args):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2seq_add = data_val
        self.u2answer = data_test
        self.max_len = args.max_len

    def get_pytorch_dataloaders(self):
        dataset = TestDataset(self.u2seq, self.u2seq_add, self.u2answer, self.max_len)
        return data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
