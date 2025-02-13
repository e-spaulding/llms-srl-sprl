# Data preparation

Some samples of data are located in the `corpus` directory, but either follow the steps below to replicate all data used in the experiments, or email `elizabeth.spaulding@colorado.edu` to skip the below steps and get the pre-processed data.

## Download the data

### Ontonotes/CoNLL 2012

Download Ontonotes 5.0 through LDC. The catalog number used for these experiments is LDC2013T19. Set an env variable for the data:

```shell
export ONTONOTES_DIR=path/to/ontonotes-release-5.0/
```

Directory structure:

```
├── ONTONOTES_DIR
│   ├── data
│   │   ├── files
│   │   │   ├── ...
│   ├── docs
│   ├── tools
│   ├── index.html
```

### PropBank Release Repo

Clone [this](https://github.com/propbank/propbank-release) repo into the data-prep directory:

```
cd data-prep
git clone git@github.com:propbank/propbank-release.git
```

Some of the scripts and data within are used for CoNLL 2012.

### SPRL

Email `elizabeth.spaulding@colorado.edu` to get the SPRL data in the needed format, and set an env variable for that data:

```shell
export SPRL_DIR=path/to/sprldata/
```

## Preprocessing script

Finally, you can execute the shell script -

```shell
data-prep/preprocess-data.sh
```

This will populate `./corpus/` with the necessary `.pkl` files for the rest of the code.