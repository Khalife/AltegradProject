def loadAll():
    with open('../data/r8_train_stemmed.txt') as lines:
        labels = []
        contents = []
        for line in lines:
            [label, content] = line.split('\t')
            labels.append(label)
            contents.append(content)

    with open('../data/r8_test_stemmed.txt') as lines:
        for line in lines:
            [label,content]=line.split(' ',1)
            labels.append(label)
            contents.append(content)

    return {"labels" : labels , "documents" : contents}