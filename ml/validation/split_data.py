
def train_test_split(data, test_size=0.2):
    ''' It works for pandas dataframes '''

    train_data = data.sample(frac=1-test_size)
    test_data = data.drop(train_data.index)

    return train_data, test_data

def train_test_validation_split(data, test_size=0.2, validation_size=0.1):
    ''' It works for pandas dataframes '''

    train_data = data.sample(frac=1-(test_size+validation_size))
    test_data = data.drop(train_data.index).sample(frac=test_size/(test_size + validation_size))
    validation_data = data.drop(train_data.index).drop(test_data.index)

    return train_data, test_data, validation_data
