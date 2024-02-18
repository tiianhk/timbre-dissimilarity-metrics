# Timbre Dissimilarity Metrics
A collection of metrics for evaluating timbre dissimilarity using the TorchMetrics API. Work in progress and subject to sudden change — use in projects at your own risk.

## Installation
`pip install -e .`

## Usage
```
import timbremetrics

datasets = timbremetrics.list_datasets()
dataset = datasets[0] # get the first timbre dataset

# MAE between target dataset and pred embedding distances
metric = timbremetrics.MAE(
    dataset=dataset, distance=timbremetrics.l1
)

# get numpy audio for the timbre dataset
audio = timbremetrics.get_audio(dataset)

# get arbitrary embeddings for the timbre dataset's audio
embeddings = net(audio)

# compute the metric
metric(embeddings)

```

## Metrics

The following metrics are implemented.

### Mean Squared Error

Gives the mean squared error between the upper triangles of the predicted distance matrix and target distance matrix:

$MSE(D_X,D_Y)=\dfrac{2}{n(n-1)}\sum\limits_{i=1}^n\sum\limits_{j=i+1}^n(D_X-D_Y)^2$

### Mean Absolute Error

Gives the mean squared error between the upper triangles of the predicted distance matrix and target distance matrix:

$MAE(D_X,D_Y)=\dfrac{2}{n(n-1)}\sum\limits_{i=1}^n\sum\limits_{j=i+1}^n|D_X-D_Y|$

### Item Rank Agreement

Gives the proportion of distances ranked per-item that match between the predicted distance matrix and target distance matrix.

$IRA(R_X,R_Y)=\dfrac{1}{n^2-n}\left[\sum\limits_{i=1}^n\sum\limits_{j=1}^n(1_{\{0\}}(R_{X_{i,j}}-R_{Y_{i,j}}))-n\right]$

Where $1_A(x)$ is the indicator function given by:

![Indicator function](https://latex.codecogs.com/png.latex?1_A%28x%29%20%3A%3D%20%5Cbegin%7Bcases%7D%201%20%5Cquad%20%5Ctext%7Bif%20%7D%20x%20%5Cin%20A%20%5C%5C%200%20%5Cquad%20%5Ctext%7Bif%20%7D%20x%20%5Cnotin%20A%20%5C%20%5Cend%7Bcases%7D)

and $R_X$ & $R_Y$ are distances matrices ranked per item such that each row contains the ordinal distances from the corresponding item. We also provide a _top-k_ version which computes this metric considering only the closest _k_ items in each row.

### Triplet Agreement

Samples pseudo-triplets from the target distance matrix according to a positivity radius and margin, and returns the proportion of these triplets for which ordering is retained in the predicted distance matrix, with the margin optionally enforced.

### Triplet K-NN Agreement
For each anchor (a) from the target distance matrix $D_X$, all triplets (a, i, j) are sampled where i and j are in a's K-nearest neighborhood and D(a, i) < D(a, j). The metric returns the proportion of these triplets for which ordering is retained in the predicted distance matrix $D_Y$. 

![all triplets](https://latex.codecogs.com/svg.image?T(M)%20=%20%5C%7B(a,%20i,%20j)%20%7C%20D(y_a,%20y_i)%20%3C%20D(y_a,%20y_j),%20i%20%5Cin%20M%20%5Cbackslash%20%5C%7Ba%5C%7D,%20j%20%5Cin%20M%20%5Cbackslash%20%5C%7Ba%5C%7D%5C%7D)

![triplet agreement](https://latex.codecogs.com/svg.image?%5Cbegin%7Bequation%7D%20%20%20%20%5Clabel%7Btriplet-agreement%7D%20%20%20%20%5Cbegin%7Baligned%7D%20%20%20%20TripletKNN%20=%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bn=1%7D%5E%7BN%7D%20%5Cmathbf%7B1%7D(a_n,%20i_n,%20j_n),%20(a_n,%20i_n,%20j_n)%20%5Cin%20T(M)%20%5C%5C%20%20%20%20%5Cmathbf%7B1%7D(a,%20i,%20j)%20=%20%20%20%20%20%5Cbegin%7Bcases%7D%20%20%20%20%20%20%20%201%20&%20D_y(a_n,%20i_n)%20%3C%20D_y(a_n,%20j_n)%5C%5C%20%20%20%20%20%20%20%200%20&%20%5Ctext%7Botherwise%7D%20%20%20%20%5Cend%7Bcases%7D%20%20%20%20%5Cend%7Baligned%7D%5Cend%7Bequation%7D)

### Mantel Test

The Mantel test computes Pearson's r or Spearman's rho on the condensed form of the upper triangles of the predicted and target distance matrices. The significance of the given result can be estimated using permutation analysis.

## References
```
@article{thoret2021learning,
  title={Learning metrics on spectrotemporal modulations reveals the perception of musical instrument timbre},
  author={Thoret, Etienne and Caramiaux, Baptiste and Depalle, Philippe and Mcadams, Stephen},
  journal={Nature Human Behaviour},
  volume={5},
  number={3},
  pages={369--377},
  year={2021},
  publisher={Nature Publishing Group}
}
```
- original data source: https://github.com/EtienneTho/musical-timbre-studies 
