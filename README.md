# II-map
This is the supplementary repo for II-map. 

The interactive UI for II-map is available at [https://patrik-ha.github.io/ii-map/](https://patrik-ha.github.io/ii-map/) for our pre-trained model.

# Reproducibility 

## Shortlist

### Common
1. Install the required Python packages. (`python -m pip install -r requirements.txt`)
2. Run [download_and_prepare_lc0_weights.ipynb](/download_and_prepare_lc0_weights.ipynb).
3. Download the [latest distributable of lc0](https://github.com/LeelaChessZero/lc0/releases/tag/v0.30.0), unzip, and place contents in your working directory in a folder called `lc0`.
4. Run `convert_to_onnx.sh`
5. Run [add_intermediate_outputs_to_onnx_models.ipynb](/add_intermediate_outputs_to_onnx_models.ipynb)
### Concept detection
1. Edit/run `python processing/concept_detection.py`
2. Plot results using [plot_concept_graphs.ipynb](/plot_concept_graphs.ipynb) for plotting results.
### GradCAM
1. Run [onnx_to_tf_model.ipynb](/onnx_to_tf_model.ipynb).
2. Run [gradcam.ipynb](/gradcam.ipynb).
### II-map
1. Run `download_data.sh`.
2. Train the model by running `python processing/ii_map.py`.
3. Get puzzle results by running [puzzles.ipynb](/puzzles.ipynb).
4. Get game results by running [lines.ipynb](/lines.ipynb).
## Detailed list

The main steps for getting and preparing networks from lc0 is a common step for both the reproducibility of the concept detection section and the II-map section.
The lc0-project has a lot of different network-types (which is all intented to be used with their [open-source software](https://github.com/LeelaChessZero/lc0) as an UCI engine), but since we are only working with networks of the same architecture as AlphaZero (20 block ResNet), the intention is to convert the network-files provided by lc0 to ONNX models that we can then use for inference for intermediate, and policy/value outputs.

### 1. Download lc0-networks
We downloaded the raw networks from [the main network page](http://training.lczero.org/networks/?show_all=1). In theory it is possible to manually find and download any network from here, but since the site lists **all** previous network iterations while making it somewhat hard to search, we extract the links to the desired files programmatically. 

This is done in the notebook [download_and_prepare_lc0_weights.ipynb](/download_and_prepare_lc0_weights.ipynb).

### 2. Convert lc0-networks to ONNX
We downloaded the [latest distributable of lc0](https://github.com/LeelaChessZero/lc0/releases/tag/v0.30.0). This has a handy routine for converting any custom lc0-network file to a standardised ONNX network.

For each network that you downloaded, run
```bash
./lc0 leela2onnx --input=network.pb --output=network.onnx
```
and put all the produced .onnx files in a folder named "prepared_networks".

### 3. Add intermediate outputs to the prepared models
We wanted to get intermediate outputs for any given model iteration, so these outputs are added to all prepared ONNX-networks.

This is done in [add_intermediate_outputs_to_onnx_models.ipynb](/add_intermediate_outputs_to_onnx_models.ipynb).

## Concept detection

### Training models
We used the prepared networks, pythonchess, and some helper code from [lczero_tools](https://github.com/so-much-meta/lczero_tools/blob/master/src/lcztools/_leela_board.py) to provide intermediate activations for all model interations for a set of [concept functions](concepts/concepts.py).

The concept detection procedure is done by running
```python
python processing/concept_detection.py
```
Results are placed in a new folder called "results".

### Producing plots
The plots shown in the paper were created by [plot_concept_graphs.ipynb](/plot_concept_graphs.ipynb)

## II-map

### Training models
We used the same pipeline as for the concept detection procedure to get pairs of positions and policy/values for training our II-map models.

To train the model itself, we retrieved a large set of positions from [a database of elite-level games](https://database.nikonoel.fr/). We specifically used the files containing games from october and november 2021. 

These pgns were combined and placed in the folder `pgns/elite.pgn`.

The training procedure is done by running
```python
python processing/ii_map.py
```
Model checkpoints are saved in ``ii_map/". We used the last checkpoint produced by this method to produce the results in our paper.

### Producing results
#### GradCAM
We produced some results applying GradCAM to chess using the best iteration of our downloaded networks. We created a Tensorflow-compatible variant of the last model iteration.

This was done by running [onnx_to_tf_model.ipynb](/onnx_to_tf_model.ipynb).

Our presented results can then be obtained by using [gradcam.ipynb](/gradcam.ipynb).

#### Puzzles
We used the [puzzle-database from lichess](https://database.lichess.org/#puzzles). We downloaded it, unzipped it (using ``unzstd"), and placed it in the working directory as "archive.csv". We then extracted some applicable FENs, and overwrote the file. This was so the file we include in the repo is a bit smaller.

This was done by using [puzzles.ipynb](/puzzles.ipynb).

#### Lines
We found PGNs for various famous games (included in this repo). 

The importance maps were created by using [lines.ipynb](/lines.ipynb).

### Frontend

This project also contains a frontend hosted for using a TF.js version of our trained masker model. 