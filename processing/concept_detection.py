import io
import os
from typing import Callable, Dict, List, Tuple, TypedDict

import chess.pgn
import numpy
import numpy as np
import onnxruntime as rt
import tensorflow.keras as keras
from leela_board import LeelaBoard
from tqdm import tqdm


def process_pgn_for_binary_concept(pgn_path: list[str], concept_function: Callable, save_as_path: str, limit=50000):
    
    with open(pgn_path) as f:
        pgns = f.read().replace("\n\n", "\n").split("\n\n")
    pgns = [p for p in pgns if "Event" not in p]

    boards_as_planes = []
    concept_outputs = []
    fens = []
    with tqdm(total=limit) as pbar:
        for pgn in pgns:
            game = chess.pgn.read_game(io.StringIO(pgn))
            if game is None:
                continue
            data_board = LeelaBoard()
            for move in game.mainline_moves():
                data_board.pc_board.push(move)
                data_board._lcz_push()
                if concept_function(data_board.pc_board) and np.random.random() > 0.5:
                    boards_as_planes.append(data_board.lcz_features())
                    fens.append(data_board.pc_board.fen())
                    concept_outputs.append(concept_function(data_board.pc_board))
                    pbar.update(1)

            if len(concept_outputs) >= limit:
                break
    boards_as_planes = np.array(boards_as_planes, dtype=np.float16)
    concept_outputs = np.array(concept_outputs, dtype=np.float16)

    os.makedirs("concept_datasets", exist_ok=True)

    np.savez(save_as_path, boards=boards_as_planes[:limit], fens=np.array(fens), concept_outputs=concept_outputs[:limit])

def find_intermediate_activations(model_string, concept_name, total_points):
    sess = rt.InferenceSession("prepared_networks/{}.onnx".format(model_string), providers=['CUDAExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    data = np.load(os.path.join("concept_datasets", concept_name + ".npz"))

    boards_as_planes = data["boards"][:total_points]
    concept_outputs = data["concept_outputs"][:total_points]

    BATCH_SIZE = 128
    intermediate_activations = [[] for _ in range(len(output_names) - 2)]
    with tqdm(total=boards_as_planes.shape[0]) as pbar:
        for i in range(0, boards_as_planes.shape[0], BATCH_SIZE):
            batch_shape = boards_as_planes[i:i+BATCH_SIZE].shape[0]
            pred = sess.run(output_names, {input_name: boards_as_planes[i:i+BATCH_SIZE].astype(numpy.float32)})
        
            policy = pred[0]
            wdl = pred[1]

            int_act = pred[2:]
            for i, l in enumerate(int_act):
                intermediate_activations[i].append(l.reshape((batch_shape, 8, 8, 256)))
            pbar.update(batch_shape)
    
    
    intermediate_activations = np.array([np.concatenate(l) for l in intermediate_activations], dtype=np.float16)
    return intermediate_activations


def binary_accuracy_with_guessing(y_true, y_pred):
    return (keras.metrics.binary_accuracy(y_true, y_pred) - 0.5) * 2

def make_probe(activation_shape, l1):
    intermediate_inputs = keras.Input(activation_shape)

    probe_output = keras.layers.Dense(1, activation="sigmoid", activity_regularizer=keras.regularizers.L1(l1))(intermediate_inputs)

    probe = keras.Model(intermediate_inputs, probe_output)

    probe.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(0.0001), metrics=[binary_accuracy_with_guessing])
    return probe

def perform_concept_detection(model_string, concept_name, TOTAL_POINTS=50000, TEST_RATIO=0.2):
    pos_acts = find_intermediate_activations(model_string, concept_name, TOTAL_POINTS)
    neg_acts = find_intermediate_activations(model_string, "control", TOTAL_POINTS)

    TEST_POINTS = int(2 * TOTAL_POINTS * (1 - TEST_RATIO))

    pos_acts = pos_acts[:TOTAL_POINTS]
    neg_acts = neg_acts[:TOTAL_POINTS]

    labels = np.array([0] * neg_acts.shape[1] + [1] * pos_acts.shape[1])

    intermediates = np.concatenate([pos_acts, neg_acts], axis=1)
    del pos_acts, neg_acts

    intermediates = np.reshape(intermediates, (intermediates.shape[0], intermediates.shape[1], np.prod(intermediates.shape[2:])))


    shuffle_indices = np.arange(len(labels))
    np.random.shuffle(shuffle_indices)

    intermediates = intermediates[:, shuffle_indices]
    labels = labels[shuffle_indices]

    lambdas = [0.01, 0.1]
    res = []
    for i in range(intermediates.shape[0]):
        lambda_results = []
        for l1 in lambdas:
            probe = make_probe(intermediates.shape[-1], l1)
            x_train, x_test, y_train, y_test = intermediates[i, :TEST_POINTS], intermediates[i, TEST_POINTS:], labels[:TEST_POINTS], labels[TEST_POINTS:]

            probe.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)

            y_pred = probe(x_test)[:, 0]
            lambda_results.append(binary_accuracy_with_guessing(y_test, y_pred))
        print("Layer {}: {}".format(i, max(lambda_results)))
        res.append(max(lambda_results))

    os.makedirs("results", exist_ok=True)

    np.save(os.path.join("results", "{}_{}.npy".format(concept_name, model_string)), np.array(res))


def perform_concept_detection_on_inputs(concept_name, TOTAL_POINTS=50000, TEST_RATIO=0.2):
    data = np.load(os.path.join("concept_datasets", concept_name + ".npz"))

    boards_as_planes = data["boards"][:TOTAL_POINTS]

    data = np.load(os.path.join("concept_datasets", "control" + ".npz"))

    random_boards = data["boards"][:TOTAL_POINTS]

    TEST_POINTS = int(2 * TOTAL_POINTS * (1 - TEST_RATIO))

    labels = np.array([0] * random_boards.shape[0] + [1] * boards_as_planes.shape[0])

    intermediates = np.concatenate([random_boards, boards_as_planes], axis=0)

    intermediates = np.reshape(intermediates, (intermediates.shape[0], np.prod(intermediates.shape[1:])))


    shuffle_indices = np.arange(len(labels))
    np.random.shuffle(shuffle_indices)

    intermediates = intermediates[shuffle_indices]
    labels = labels[shuffle_indices]

    lambdas = [0.01, 0.1]
    res = []
    lambda_results = []
    for l1 in lambdas:
        probe = make_probe(intermediates.shape[-1], l1)
        x_train, x_test, y_train, y_test = intermediates[:TEST_POINTS], intermediates[TEST_POINTS:], labels[:TEST_POINTS], labels[TEST_POINTS:]

        probe.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15, batch_size=32)

        y_pred = probe(x_test)[:, 0]
        lambda_results.append(binary_accuracy_with_guessing(y_test, y_pred))
    print("Inputs {}".format(max(lambda_results)))
    res.append(max(lambda_results))

    os.makedirs("results", exist_ok=True)

    np.save(os.path.join("results", "{}_inputs.npy".format(concept_name)), np.array(res))


from concepts import *

concepts = [
    ("threat_my_queen", threat_my_queen),
    ("threat_opp_queen", threat_opp_queen),
    ("has_mate_threat", has_mate_threat),
    ("in_check", in_check),
    ("has_contested_open_file", has_contested_open_file),
    ("has_own_double_pawn", has_own_double_pawn),
    ("has_opp_double_pawn", has_opp_double_pawn),
    ("material_advantage", material_advantage),
    ("random", random)
]
concept_path = os.path.join("concept_datasets", "control" + ".npz")
if not os.path.exists(concept_path):
    print("Creating concept dataset for {}".format("control"))
    process_pgn_for_binary_concept("pgns/elite.pgn", random, concept_path)

# for CONCEPT_NAME, concept_func in concepts:
#     # perform_concept_detection_on_inputs(CONCEPT_NAME)
#     print("Probing for {}".format(CONCEPT_NAME))
#     concept_path = os.path.join("concept_datasets", CONCEPT_NAME + ".npz")
#     if not os.path.exists(concept_path):
#         print("Creating concept dataset for {}".format(CONCEPT_NAME))
#         process_pgn_for_binary_concept("pgns/elite.pgn", concept_func, concept_path)
#     model_strings = os.listdir("prepared_networks")
#     model_strings = [m.replace(".onnx" , "") for m in model_strings]

#     for m in model_strings:
#         print(m)
#         perform_concept_detection(m, CONCEPT_NAME, TOTAL_POINTS=50000)


# perform_concept_detection("30001", "has_mate_threat", TOTAL_POINTS=50000)