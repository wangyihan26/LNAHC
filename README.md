<meta name="robots" content="noindex">

# Source Code for LNAHC

Code: code is available under the root folder.<br/>
Data: data is available under the folder dataset/.<br/>

----------------------------------------------
### Requirements:<br/>
Python 3.9 (Recommended)<br/>
PyTorch 2.5.1+cu121<br/>
NumPy 1.24.3<br/>
SciPy 1.10.1<br/>
Scikit-learn 1.6.1<br/>
PyYAML 6.0.2<br/>

----------------------------------------------
### Usage:<br/>
Run the proposed method LNAHC by executing:<br/>
python LNAHC.py --dataset cocitation.cora <br/>
The --dataset argument should be one of the available datasets.<br/>

Available datasets configured in `config.yaml` include:<br/>
- cocitation/cora, cocitation/citeseer, cocitation/pubmed<br/>
- coauthorship/cora, coauthorship/dblp<br/>
- zoo, 20newsW100, Mushroom, NTU2012, ModelNet40<br/>

Other parameters are optional and controlled via the configuration file:<br/>
--config: Path to the YAML configuration file. The default value is `config.yaml`.<br/>

Hyperparameters can be modified directly in `config.yaml` for each dataset. Key parameters include:<br/>
- rw_steps: The number of steps for the dual-channel propagation.<br/>
- alpha: The restart probability in the structural random walk.<br/>
- beta: The balance factor between structural and semantic features.<br/>
- gamma_ncut: The weight for the refined Attribute-aware Hypergraph Ncut loss.<br/>
- kl: The weight for the cluster-level reconstruction loss.<br/>

----------------------------------------------
### Examples:<br/>

Example-1:<br/>
Evaluate the clustering results of LNAHC on dataset cocitation/cora<br/>
```bash
python LNAHC.py --dataset cocitation/cora