**GRTr**: **G**enerative-**R**etrieval **Tr**ansformers
==

Code for the paper: [Igor Shalyminov](https://ishalyminov.github.io/), [Alessandro Sordoni](https://www.microsoft.com/en-us/research/people/alsordon/), [Adam Atkinson](https://www.microsoft.com/en-us/research/people/adatkins/), [Hannes Schulz](https://www.microsoft.com/en-us/research/people/haschulz/). "Hybrid Generative-Retrieval Transformers for Dialogue Domain Adaptation".

Installation
------------

```
$ cd code-directory
$ conda create -n grtr python=3.7 cython
$ conda activate grtr
$ pip install -e .
```

For mixed precision training:

```bash
$ pip install git+https://github.com/nvidia/apex
```
 
Training a base GPT-2 model on MetaLWOz
--------

```bash
$ python scripts/train ~/data/blobfuse/mldc/metalwoz/dataset/metalwoz-v1.zip ~/data/blobfuse/mldc/metalwoz_dataspec.json --dataset_cache cache exp/grtr --train_batch_size 4 --valid_batch_size 4 --early_stopping_after -1 --n_epochs 25
```

Predictions
-----------

- generate-and-rank

```sh
python scripts/predict_generate_and_rank <MetaLWOz/MultiWoz zipfile> <testspec json> <output dir> <base GPT-2 model dir> --fine-tune --dataset_cache cache exp/grtr --train_batch_size 4 --valid_batch_size 4
```

- generate only

```sh
python scripts/predict <MetaLWOz/MultiWoz zipfile> <testspec json> <output dir> <base GPT-2 model dir> --fine-tune --dataset_cache cache exp/grtr --train_batch_size 4 --valid_batch_size 4
```


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
