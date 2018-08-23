
def accuracy(model, data, result_label='label'):
    good, bad = 0, 0

    for index, row in data.iterrows():
        if model.predict(row.to_dict()) == row[result_label]:
            good += 1
        else:
            bad += 1

    return good/(good + bad)
