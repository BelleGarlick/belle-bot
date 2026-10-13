def get_infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch
