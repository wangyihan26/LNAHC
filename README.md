<meta name="robots" content="noindex">

# LNAHC: Lightweight Neural Attributed Hypergraph Clustering via Multi-hop Attributed HNCut

In this paper, we propose LNAHC (Lightweight Neural Attributed Hypergraph Clustering) , a lightweight, end-to-end neural framework for attributed hypergraph clustering. Its main idea is to seamlessly unify representation learning and clustering optimization , and specifically, it introduces an attribute-aware hypergraph random walk strategy to construct a structure-attribute proximity matrix that preserves both local topology and multi-hop global semantic relationships. Together with a novel differentiable attributed hypergraph normalized cut (AHNCut) module that directly optimizes a relaxed cut-like loss, these components enable LNAHC to regularize the embedding space and produce high-quality clusters without relying on computationally intensive contrastive learning, heavy attention mechanisms, or decoupled post-processing steps. Our extensive experiments using 9 state-of-the-art baselines across 7 real-world datasets demonstrate the effectiveness and efficiency of LNAHC, showing that it consistently achieves superior clustering performance while maintaining a highly lightweight architecture.

Code: code is available under the root folder.<br/>
Data: data is available under the folder dataset/.<br/>

----------------------------------------------
### Requirements:<br/>
Python 3.9 <br/>
PyTorch 2.5.1+cu121<br/>
NumPy 1.24.3<br/>
SciPy 1.10.1<br/>
Scikit-learn 1.6.1<br/>
PyYAML 6.0.2<br/>

For ease of setup, you can install all required dependencies in one go using the provided `requirements.txt` file:
```bash
   pip install -r requirements.txt
```

----------------------------------------------
### Usage

To run the LNAHC clustering algorithm on a dataset, please specify the target dataset using the command-line parameter `--dataset`. 

The datasets supported by our implementation (and pre-configured in `config.yaml`) include the following:

| Type of dataset | `--dataset` argument |
| :--- | :--- |
| **Co-citation hypergraphs** | `cocitation/cora`, `cocitation/citeseer`, `cocitation/pubmed` |
| **Other Attributed hypergraphs** | `20newsW100`, `Mushroom`, `NTU2012`, `ModelNet40` |

Other command-line parameters are optional:

| Parameter  | Default       | Description                                                                      |
| :---       | :---          |:---------------------------------------------------------------------------------|
| `--config` | `config.yaml` | Path to the YAML configuration file containing dataset-specific hyperparameters. |

---

### Configuration File (`config.yaml`)

Instead of passing dozens of arguments through the command line, LNAHC utilizes a centralized `config.yaml` file to manage dataset-specific hyperparameters. The script dynamically loads the optimal settings for the chosen dataset at runtime. 

You can easily test different configurations by modifying the values directly in `config.yaml`. Below is a detailed description of each hyperparameter used in our model:

| Parameter    | Description                                                                                        |
|:-------------|:---------------------------------------------------------------------------------------------------| 
| `hidden`     | The dimension size of the hidden layers in the Hypergraph Neural Network (HGNN).                   |
| `dropout`    | Dropout rate used in the HGNN layers to prevent overfitting.                                       |
| `lr`         | Learning rate for the Adam optimizer.                                                              |
| `epochs`     | The total number of training epochs.                                                               |
| `rw_steps`   | The number of propagation steps in the dual-channel hypergraph random walk prior learner.          |
| `alpha`      | $\alpha$, the restart/teleport probability in the muti-hop hypergraph random walk.                 |
| `beta`       | $\beta$, the balance factor controlling the weight between structural and semantic features.       |
| `decay`      | L2 regularization (weight decay) factor for the optimizer.                                         |
| `ncut`       | $L_{cluster}$,the weight coefficient for the refined AHNcut loss.                                  |
| `balance`    | $L_{balance}$, the weight coefficient for the balance regularization prevents degenerate solution. |
| `gamma_nrec` | $\gamma_{nrec}$, the weight coefficient for the node-level reconstruction loss.                    |
| `gamma_crec` | $\gamma_{crec}$, the weight coefficient for the cluster-level reconstruction loss.                 |                                                           |


---

### Reproducing Results

To easily reproduce the results presented in our paper, please use the following commands for the corresponding datasets. The script will automatically load the optimal hyperparameters from `config.yaml`.

###### Co-citation Cora
`python LNAHC.py --dataset cocitation/cora`

###### Co-citation Citeseer
`python LNAHC.py --dataset cocitation/citeseer`

###### Co-citation Pubmed
`python LNAHC.py --dataset cocitation/pubmed`

###### 20News
`python LNAHC.py --dataset 20newsW100`

###### Mushroom
`python LNAHC.py --dataset Mushroom`

###### NTU2012
`python LNAHC.py --dataset NTU2012 `

###### ModelNet40
`python LNAHC.py --dataset ModelNet40`

Sample output of LNAHC, on Co-citation Cora dataset:
```
Final Epoch | ACC: 0.7253 | NMI: 0.5652 | F1: 0.6930 | ARI: 0.5031
Effective Time: 13.79s
```
