
def random_sampling(k, data=None):
    '''Compatible with dataframes of pandas'''
    if data == None:
        def sampling(data):
            m = k
            if k > data.shape[0]:
                m = data.shape[0]
            return data.sample(n=m)
        return sampling
    else:
        if k > data.shape[0]:
            k = data.shape[0]
        return data.sample(n=k)


