import argparse
import yaml
import numpy as np

'''
https://github.com/yuleiniu/rva/blob/visdialch/data/init_glove.py
'''

def loadGloveModel(gloveFile):
    print("Loading pretrained word vectors...")
    with open(gloveFile,'r') as f:
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

def init_glove(tokenizer):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-yml", default="configs/rva.yml",
        help="Path to a config file listing reader, model and solver parameters."
    )
    parser.add_argument(
        "--pretrained-txt", default="tasks/SCoA/data/glove.6B.300d.txt",
        help="Path to GloVe pretrained word vectors."
    )
    parser.add_argument(
        "--save-npy", default="tasks/SCoA/data/glove.npy",
        help="Path to save word embeddings."
    )
    args = parser.parse_args(args=[])

    glove = loadGloveModel(args.pretrained_txt)

    vocabulary = tokenizer.vocab
    vocab_size = len(vocabulary)

    glove_data = np.zeros(shape=[vocab_size, 300], dtype=np.float32)
    for i, word in enumerate(vocabulary):
        if word in ['<PAD>', '<UNK>', '<EOS>', '<NAV>', '<ORA>', '<TAR>', '<OBJ>', '<DIR>', '<ACT>']:
            continue
        if word in glove:
            glove_data[i] = glove[word]
        else:
            glove_data[i] = glove['unk']
    np.save(args.save_npy, glove_data)

    print("Finish glove")
