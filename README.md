# Phosphosite Prediction with Linear SVMs

A reproducible machine-learning pipeline for predicting human protein phosphorylation sites from local amino-acid sequence context.

This project compares two representations of 21-residue windows centred on serine (S), threonine (T), or tyrosine (Y):

- **Position-specific one-hot encoding + linear SVM**
- **ESM-2 protein language model embeddings + linear SVM**

S/T sites and Y sites are modelled separately. Cross-validation is grouped by UniProt accession so that windows from the same protein cannot appear in both training and validation folds.

## Project overview

The workflow was developed during a research placement in the **Child Lab, Imperial College London**. The original experimental scripts have been reorganised here into a clearer, reproducible repository structure.

The main modelling question was whether contextual representations learned by a protein language model improve phosphosite classification relative to a simple position-specific sequence baseline.

## Workflow

```text
Positive phosphosite table
        |
        v
Retrieve UniProt sequences + extract 21-aa windows
        |
        v
Filter canonical, valid S/T/Y-centred positives
        |
        +------------------------------+
        |                              |
        v                              v
Human proteome FASTA              Positive windows
        |                              |
        v                              |
Generate unannotated S/T/Y ------------+
putative negatives                     |
        |                              |
        +--------------+---------------+
                       |
                       v
              Build modelling dataset
                       |
             +---------+---------+
             |                   |
             v                   v
       One-hot encoding       ESM-2 embeddings
             |                   |
             v                   v
        Linear SVM            Linear SVM
             |                   |
             +---------+---------+
                       |
                       v
          Protein-grouped cross-validation
```

## Repository structure

```text
phosphosite-svm/
├── README.md
├── environment.yml
├── main.nf
├── nextflow.config
├── .gitignore
│
├── src/
│   ├── data/
│   │   ├── extract_positive_windows.py
│   │   ├── filter_positive_sites.py
│   │   ├── build_negative_sites.py
│   │   └── build_training_dataset.py
│   │
│   ├── features/
│   │   └── generate_esm2_embeddings.py
│   │
│   └── models/
│       ├── train_onehot_svm.py
│       └── train_plm_svm.py
│
├── data/
│   ├── raw/
│   └── processed/
├── results/
└── models/
```

## Data preparation

### 1. Positive phosphosite windows

`src/data/extract_positive_windows.py` retrieves protein sequences from UniProt and extracts ±10 residues around each supplied phosphosite, generating a 21-residue local sequence window where possible.

`src/data/filter_positive_sites.py` then retains canonical, error-free S/T/Y-centred sites and removes duplicate protein-position pairs.

### 2. Putative negative sites

`src/data/build_negative_sites.py` reads a human proteome FASTA and enumerates S/T/Y residues in proteins represented in the positive dataset. Known positive positions are excluded.

By default, residues too close to the termini to provide a complete 21-aa window are omitted.

> **Important:** these are *putative negatives*. An unannotated residue is not necessarily biologically incapable of phosphorylation.

### 3. Final modelling dataset

`src/data/build_training_dataset.py` converts positive and negative examples into a common schema, validates coordinates and central residues, removes duplicates and positive/negative overlaps, and writes the final binary-labelled modelling table.

## Models

### One-hot linear SVM

`src/models/train_onehot_svm.py` creates a sparse position-specific representation of each 21-aa sequence window and trains `LinearSVC` classifiers.

Key design choices:

- separate **ST** and **Y** models
- `class_weight="balanced"` for class imbalance
- 5-fold `StratifiedGroupKFold`
- grouping by **protein accession** to reduce protein-level information leakage
- hyperparameter selection over the SVM regularisation parameter `C`

The code reports:

- PR-AUC
- ROC-AUC
- Matthews correlation coefficient (MCC)
- precision
- recall
- F1 score
- balanced accuracy

### ESM-2 embeddings + linear SVM

`src/features/generate_esm2_embeddings.py` uses:

```text
facebook/esm2_t6_8M_UR50D
```

For each 21-aa window, the representation concatenates:

1. the embedding of the central phospho-acceptor residue; and
2. the mean embedding across the 21 residues.

For this ESM-2 model, this produces a **640-dimensional feature vector**.

`src/models/train_plm_svm.py` evaluates linear SVMs on these embeddings using the same protein-grouped cross-validation strategy.

## Preliminary result

During the placement, the downsized training experiments produced **MCC values of approximately 0.38** for the linear models. This repository focuses on the reproducible analysis code rather than claiming a definitive benchmark; exact run-level metrics should be taken from the associated experiment outputs when available.

## Installation

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate phosphosite-svm
```

## Run with Nextflow

The repository includes a Nextflow wrapper to make the reorganised pipeline easier to reproduce.

```bash
nextflow run main.nf \
  --positive_xlsx path/to/positive_sites.xlsx \
  --human_fasta path/to/human_proteome.fasta \
  --uniprot_col uniprot \
  --position_col position
```

Nextflow was added when the research code was reorganised for reproducibility; it should not be interpreted as implying that every original experiment was initially executed through Nextflow.

## MLflow tracking

The model-training scripts support MLflow experiment tracking. Without an external tracking URI, runs are stored locally.

To connect to a local MLflow server:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
mlflow server --host 127.0.0.1 --port 5000
```

## Dataset source

The positive phosphosite data used in this work were derived from the human phosphoproteome resource described in:

> Ochoa D, Jarnuczak AF, Viéitez C, *et al.* **The functional landscape of the human phosphoproteome.** *Nature Biotechnology* 38, 365–373 (2020). DOI: `10.1038/s41587-019-0344-3`.

The study assembled a reference human phosphoproteome from large-scale phosphoproteomics datasets. Raw and intermediate data from the original study are available through PRIDE under accession **PXD012174**.

Large datasets, proteome FASTA files, generated embeddings, trained models, and MLflow artefacts are intentionally excluded from this repository.

## Reproducibility notes

The repository is a cleaned and reorganised version of the experimental analysis code. The scientific logic has been preserved, while file naming and workflow organisation have been simplified for readability and reproducibility.

The negative-class construction is a major limitation of the task: lack of phosphorylation annotation does not guarantee a true biological negative. Consequently, model performance should be interpreted as discrimination between annotated positives and unannotated candidate sites under this dataset construction procedure.

## Skills demonstrated

This project demonstrates practical use of:

`Python` · `pandas` · `NumPy` · `scikit-learn` · `PyTorch` · `Transformers` · `ESM-2` · `SVM` · `MLflow` · `Nextflow` · `UniProt REST API`
