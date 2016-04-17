import DataSet

if __name__ == '__main__':
    dataset = DataSet(5, 10)
    dataset.load_data('train', 10, '')
    dataset.train()