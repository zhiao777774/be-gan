import torch.utils.data


class BaseDest(torch.utils.data.Dataset):
    def __init__(self):

        pass

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

    def norm(self, batch_size, num_workers, length=0.8):
        train_size = int(length * len(self.data))
        test_size = len(self.data) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(self.data, [train_size, test_size])

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

