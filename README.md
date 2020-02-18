**GRTr**: **G**enerative-**R**etrieval **Tr**ansformers
==

Code for the paper ["Hybrid Generative-Retrieval Transformers for Dialogue Domain Adaptation"](https://drive.google.com/file/d/19ifYuZofZMslocQzhICgGXTHFk90b58f/view).

By [Igor Shalyminov](https://ishalyminov.github.io/), [Alessandro Sordoni](https://www.microsoft.com/en-us/research/people/alsordon/), [Adam Atkinson](https://www.microsoft.com/en-us/research/people/adatkins/), [Hannes Schulz](https://www.microsoft.com/en-us/research/people/haschulz/). 

Installation
------------

```bash
$ cd code-directory
$ git submodule init
$ git submodule update
$ conda create -n hybrid_retgen python=3.7
$ conda activate hybrid_retgen
$ conda install cython
$ pip install -e ./dstc8-metalearn-baseline
$ pip install -e .
```

For mixed precision training:

```bash
$ pip install git+https://github.com/nvidia/apex
```
 
Training a base GPT-2 model on MetaLWOz
--------

```bash
$ python scripts/train <MetaLWOz zipfile> metalwoz_dataspec.json --dataset_cache cache exp/grtr --train_batch_size 4 --valid_batch_size 4 --early_stopping_after -1 --n_epochs 25
```

Add `--fp16 O1` to use mixed precision training.

Predictions
-----------

### generate-and-rank

```sh
python scripts/predict_generate_and_rank <MetaLWOz/MultiWoz zipfile> <testspec json> <output dir> <base GPT-2 model dir> --fine-tune --dataset_cache cache exp/grtr --train_batch_size 4 --valid_batch_size 4
```

### generate only

```sh
python scripts/predict <MetaLWOz/MultiWoz zipfile> <testspec json> <output dir> <base GPT-2 model dir> --fine-tune --dataset_cache cache exp/grtr --train_batch_size 4 --valid_batch_size 4
```

Convenience `bash` scripts are provided in `scripts/` to produce predictions for each of the three test specs.

Evaluation can be done using the [evaluate script](https://github.com/microsoft/dstc8-meta-dialog/blob/master/scripts/evaluate) in the competition baseline repository.

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
