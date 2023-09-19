import io
import os
from random import shuffle

import chess.pgn
import larq
import numpy
import numpy as np
import onnxruntime as rt
import tensorflow as tf
import tensorflow.keras as keras
from leela_board import LeelaBoard
from tqdm import tqdm


class RandomInputs(tf.keras.layers.Layer):
    def __init__(self, output_res, **kwargs):
        super(RandomInputs, self).__init__(**kwargs)
        self.output_res = output_res
    
    def build(self, input_shapes):
        super(RandomInputs, self).build(input_shapes) 
        pass

    def call(self, inputs, training=None):
        return tf.random.uniform(self.output_res, 0, 1)

def softmax(x):
    mx = np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(x - mx)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    return numerator / denominator

def make_student_model(BLOCK_COUNT=20, FILTER_SIZE=256):
    keras.backend.set_image_data_format('channels_first')

    inputs = keras.Input((20, 8, 8))
    slice = keras.layers.Lambda(lambda w: w[:, :12, :, :])(inputs)
    remaining_slice = keras.layers.Lambda(lambda w: w[:, 12:, :, :])(inputs)

    base = keras.layers.Conv2D(64, (3, 3), activation="elu", padding="same", name="res_block_output_base_mask")(inputs)
    block_amount = 7
    for block in range(block_amount):
        # Convolve "input" twice
        conv = keras.layers.Conv2D(64, (3, 3), activation="elu", padding="same")(base)
        # Add and relu residues to the doubly convolved layer
        conv_with_residues = keras.layers.Add()([base, conv])
        base = keras.layers.ELU(name="res_block_output_{}_mask".format(block))(conv_with_residues)
    heatmap = keras.layers.Conv2D(12, (1, 1), activation="sigmoid", activity_regularizer=tf.keras.regularizers.L1(0.00005), padding="same", name="heatmap")(base)

    randoms = RandomInputs((12, 8, 8))([])
    BIAS_AID = 0.0


    def add_and_reduce(w):
        random_inputs, mask = w
        return mask - random_inputs + BIAS_AID

    add = keras.layers.Lambda(add_and_reduce, name="add_reduce")([randoms, heatmap])
    mask = tf.keras.layers.Activation("ste_heaviside", name="mask")(add)


    def mask_away(w):
        mask_away, inputs = w
        return tf.math.multiply(mask_away, inputs)

    masked_input = tf.keras.layers.Lambda(mask_away, name="mult_mask")([mask, slice])
    masked_input = tf.keras.layers.Concatenate(axis=1)([masked_input, remaining_slice])


    def block(last, i):
        x = keras.layers.Conv2D(FILTER_SIZE, (3, 3), padding="same", name="block{}/conv{}".format(i, 1), activation="relu")(last)
        x = keras.layers.Conv2D(FILTER_SIZE, (3, 3), padding="same", name="block{}/conv{}".format(i, 2), activation="linear")(x)
        add = keras.layers.Add()([x, last])
        rel = keras.layers.ELU()(add)
        return rel, add


    last = keras.layers.Conv2D(FILTER_SIZE, (3, 3), padding="same", name="inputconv")(masked_input)
    last = keras.layers.ELU()(last)
    for b in range(0, BLOCK_COUNT):
        last, inter = block(last, b)

    pol_conv = keras.layers.Conv2D(FILTER_SIZE, (1, 1), padding="same", activation="relu", name="pol_conv0")(last)    
    pol_conv = keras.layers.Conv2D(8, (1, 1), padding="same", activation="linear", name="pol_conv1")(pol_conv)    
    pol_conv = keras.layers.Flatten()(pol_conv)
    pol_conv = keras.layers.Dense(1858, activation="elu", name="pol_conv3")(pol_conv)    
    pol_output = keras.layers.Softmax(name="pol_out")(pol_conv)

    val_conv = keras.layers.Conv2D(1, (1, 1), padding="same", activation="relu" , name="val_conv0")(last)    
    val_conv = keras.layers.Flatten()(val_conv)
    val_conv = keras.layers.Dense(256, activation="elu", name="val_linear")(val_conv)    
    val_output = keras.layers.Dense(1, activation="tanh", name="val_out")(val_conv)   

    mask_utilisation = tf.reduce_sum(tf.clip_by_value(tf.reduce_sum(heatmap, axis=1), 0, 1)) / (8 * 8 * 512)

    model = keras.Model(inputs, [pol_output, val_output])
    model.add_metric(mask_utilisation, name="mask_util")
    print(model.summary())
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00005), loss=[keras.losses.CategoricalCrossentropy(), keras.losses.MeanSquaredError()], loss_weights=[1.0, 0.01])


    return model


model_name = "32390"

sess = rt.InferenceSession("prepared_networks/{}.onnx".format(model_name), providers=['CUDAExecutionProvider'])
input_name = sess.get_inputs()[0].name
# output_names = [o.name for o in sess.get_outputs()]
output_names = ['/output/policy', '/output/value']

model = make_student_model()

def get_training_block(pgns, index, fold_count):
    import gc
    
    pgns_per_block = int(len(pgns) / fold_count)
    pgns_for_block = pgns[pgns_per_block * index:pgns_per_block * (index + 1)]


    boards_as_planes = []
    for pgn in tqdm(pgns_for_block, total=len(pgns_for_block)):
        game = chess.pgn.read_game(io.StringIO(pgn))
        if game is not None:
            data_board = LeelaBoard()
            for move in game.mainline_moves():
                data_board.pc_board.push(move)
                data_board._lcz_push()
                boards_as_planes.append(data_board.lcz_features())
        else:
            print("Skipped one...")

    boards_as_planes = np.array(boards_as_planes, dtype=np.float16)

    policies = np.zeros((boards_as_planes.shape[0], 1858), dtype=np.float16)
    q = np.zeros((boards_as_planes.shape[0], 1), dtype=np.float16)

    for i in tqdm(range(0, boards_as_planes.shape[0], 128), total=boards_as_planes.shape[0] // 128):
        pred = sess.run(output_names, {input_name: boards_as_planes[i:i+128].astype(numpy.float32)})
        policies[i:i+128] = softmax(pred[0])
        q[i:i+128] = pred[1]

    gc.collect()
    return boards_as_planes[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -8, -7, -6, -5, -4, -3, -2, -1]], policies, q


reps = 1
folds = 600
module_out_path = "ii_map_models"
pgn_path = "pgns/elite.pgn"

with open(pgn_path) as f:
    all_pgns = f.read().replace("\n\n", "\n").split("\n\n")
all_pgns = [p for p in all_pgns if "Event" not in p]


os.makedirs("ii_map_models", exist_ok=True)

shuffle(all_pgns)

for rep in range(reps):
    for fold in range(folds):
        # if fold == 0:
        #     model = make_student_model()
        # else:
        #     model = keras.models.load_model("distilled/epoch_{}".format(fold - 1))

        print("Getting block for fold {}, rep {}".format(fold, rep))
        boards, pols, q = get_training_block(all_pgns, fold, folds)

        # pols = softmax(pols)
        with tf.device('/CPU:0'):
            boards = tf.constant(boards)
            pols = tf.constant(pols)
            q = tf.constant(q)
        
        print("Block ready!")

        model.fit(boards, [pols, q], epochs=1, batch_size=512)
        if fold == 200 or fold == 400:
            model.save("ii_map_models/epoch_{}_rep_{}".format(fold, rep))

        # keras.backend.clear_session()

        # del model
        # gc.collect()
    model.save("ii_map_models/complete_rep_{}".format(rep))