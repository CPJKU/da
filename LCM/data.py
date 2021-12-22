from torch.utils.data import Dataset as TorchDataset
import random


# collecting Dictionary data containing actual sequential data, labels and domain labels
class ProcessedDataset(TorchDataset):
    def __init__(self, tensors):
        self.data = tensors['data']
        self.labels = tensors['labels']
        self.domain_labels = tensors['domain_labels'].long()

    def __getitem__(self, index):
        return self.data[index], self.domain_labels[index], self.labels[index]

    def __len__(self):
        return len(self.data)


# to balance target domain and source domain, oversampling can be used
class OversamplingDataset(TorchDataset):
    def __init__(self, dataset, oversampling_rate, oversampling_domain=1):
        self.dataset = dataset
        self.available_indexes = list(range(len(self.dataset)))
        if oversampling_rate:
            oversampling_indices = []
            for i in range(len(self.dataset)):
                domain_label = self.dataset[i][1]
                if domain_label == oversampling_domain:
                    oversampling_indices.append(i)
            oversampling_indices = oversampling_indices * oversampling_rate
            self.available_indexes.extend(oversampling_indices)
            random.shuffle(self.available_indexes)

        print("Len oversampled dataset: ", len(self))

    def __getitem__(self, index):
        return self.dataset[self.available_indexes[index]]

    def __len__(self):
        return len(self.available_indexes)


# creates each batch according to same amount of source and target domain data
class EqualSamplingDataset(TorchDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.source_idx_pool = list()
        self.target_idx_pool = list()
        for i in range(len(self.dataset)):
            domain_label = self.dataset[i][1]
            if domain_label == 0:
                self.source_idx_pool.append(i)
            elif domain_label == 1:
                self.target_idx_pool.append(i)

        self.ret_src = True
        self.src_pool = self.source_idx_pool.copy()
        self.tgt_pool = self.target_idx_pool.copy()

    def __getitem__(self, item):
        if self.ret_src:
            self.ret_src = not self.ret_src
            if len(self.src_pool) == 0:
                self.src_pool = self.source_idx_pool.copy()
            index = random.choice(self.src_pool)
            self.src_pool.remove(index)
            return self.dataset[index]
        else:
            self.ret_src = not self.ret_src
            if len(self.tgt_pool) == 0:
                self.tgt_pool = self.target_idx_pool.copy()
            index = random.choice(self.tgt_pool)
            self.tgt_pool.remove(index)
            return self.dataset[index]

    def __len__(self):
        return 2 * len(self.source_idx_pool)
