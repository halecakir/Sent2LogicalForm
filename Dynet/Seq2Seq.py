# files
TEST_FILE = "data/test.txt"
TRAIN_FILE = "data/train.txt"
F_VOCAB_FILE = "data/vocab.f.txt"
Q_VOCAB_FILE = "data/vocab.q.txt"


import dynet_config

# Declare GPU as the default device type
dynet_config.set_gpu()
# Set some parameters manualy
dynet_config.set(mem=400, random_seed=123456789)
# Initialize dynet import using above configuration in the current scope
import dynet as dy


from mnnl import RNNSequencePredictor
import random

random.seed(33)


def read_data(fh):
    for line in fh:
        sentence, lf = line.strip().split("\t")
        sentence = sentence.split()
        lf = lf.split()
        yield sentence, lf


def read_vocab(filename):
    t2i = {"_UNK": 0, "<s>": 1, "</s>": 2}
    with open(filename) as target:
        for line in target:
            token = line.strip().split()[0]
            if token not in t2i:
                t2i[token] = len(t2i)
    return t2i


class Seq2Seq:
    def __init__(self, w2i, lf2i, options):
        self.options = options
        self.w2i = w2i
        self.lf2i = lf2i
        self.wdims = options.wembedding_dims
        self.lfdims = options.lfembedding_dims
        self.ldims = options.lstm_dims
        self.ext_embeddings = None

        self.model = dy.ParameterCollection()
        self.trainer = dy.AdamTrainer(self.model)
        self.__load_model()
        self.wlookup = self.model.add_lookup_parameters((len(w2i), self.wdims))
        self.lflookup = self.model.add_lookup_parameters((len(lf2i), self.lfdims))

        self.context_encoder = [
            dy.VanillaLSTMBuilder(1, self.wdims, self.ldims, self.model)
        ]
        self.logical_form_decoder = dy.VanillaLSTMBuilder(
            1, self.lfdims, self.ldims, self.model
        )

        self.W_s = self.model.add_parameters((len(self.lf2i), self.ldims))
        self.W_sb = self.model.add_parameters((len(self.lf2i)))

    def __load_model(self):
        if self.options.external_embedding is not None:
            if os.path.isfile(
                os.path.join(
                    self.options.saved_parameters_dir, self.options.saved_prevectors
                )
            ):
                self.__load_external_embeddings(
                    os.path.join(
                        self.options.saved_parameters_dir, self.options.saved_prevectors
                    ),
                    "pickle",
                )
            else:
                self.__load_external_embeddings(
                    self.options.external_embedding,
                    self.options.external_embedding_type,
                )
                self.__save_model()

    def __save_model(self):
        IOUtils.save_embeddings(
            os.path.join(
                self.options.saved_parameters_dir, self.options.saved_prevectors
            ),
            self.ext_embeddings,
        )

    def __load_external_embeddings(self, embedding_file, embedding_file_type):
        ext_embeddings, ext_emb_dim = IOUtils.load_embeddings_file(
            embedding_file, embedding_file_type, lower=True
        )
        assert ext_emb_dim == self.wdims
        self.ext_embeddings = {}
        print("Initializing word embeddings by pre-trained vectors")
        count = 0
        for word in self.w2i:
            if word in ext_embeddings:
                count += 1
                self.ext_embeddings[word] = ext_embeddings[word]
                self.wlookup.init_row(self.w2i[word], ext_embeddings[word])
        print(
            "Vocab size: %d; #words having pretrained vectors: %d"
            % (len(self.w2i), count)
        )

    def train(self, train_path):
        with open(train_path, "r") as train:
            shuffledData = list(read_data(train))
            random.shuffle(shuffledData)

            for iPair, (sentence, lf) in enumerate(shuffledData):
                print(iPair, sentence)
                # I-Context Encoding
                lstm_forward = self.context_encoder[0].initial_state()

                for entry in sentence:
                    lstm_forward = lstm_forward.add_input(
                        self.wlookup[
                            self.w2i[entry] if entry in self.w2i else self.w2i["_UNK"]
                        ]
                    )
                hidden_context = lstm_forward.h()
                init_h = [dy.ones(self.ldims)]
                state = self.logical_form_decoder.initial_state()
                state.set_h(init_h)

                dy.renew_cg()


class Options:
    def __init__(self):
        self.wembedding_dims = 300
        self.lfembedding_dims = 64
        self.lstm_dims = 128
        self.external_embedding = None


w2i = read_vocab(Q_VOCAB_FILE)
lf2i = read_vocab(F_VOCAB_FILE)

options = Options()
model = Seq2Seq(w2i, lf2i, options)

model.train(TEST_FILE)
