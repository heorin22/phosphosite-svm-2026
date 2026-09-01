nextflow.enable.dsl=2

params.positive_xlsx = null
params.human_fasta   = null
params.uniprot_col   = 'uniprot'
params.position_col  = 'position'
params.outdir        = 'results'

if (!params.positive_xlsx || !params.human_fasta) {
    error "Usage: nextflow run main.nf --positive_xlsx <file.xlsx> --human_fasta <proteome.fasta>"
}

process EXTRACT_POSITIVE_WINDOWS {
    tag 'positive windows'

    input:
    path positive_xlsx

    output:
    path 'positive_windows_all.csv', emit: all_windows

    script:
    """
    python src/data/extract_positive_windows.py \
      --input ${positive_xlsx} \
      --output positive_windows_all.csv \
      --window 10 \
      --uniprot-col ${params.uniprot_col} \
      --position-col ${params.position_col}
    """
}

process FILTER_POSITIVE_SITES {
    tag 'filter positive sites'

    input:
    path all_windows

    output:
    path 'functional_score_with_regions_valid.csv', emit: valid_positive
    path 'positive_excluded_rows.csv', emit: excluded_positive

    script:
    """
    python src/data/filter_positive_sites.py \
      --input ${all_windows} \
      --valid-output functional_score_with_regions_valid.csv \
      --excluded-output positive_excluded_rows.csv
    """
}

process BUILD_NEGATIVE_SITES {
    tag 'negative sites'

    input:
    path valid_positive
    path human_fasta

    output:
    path 'negative_phosphosites_with_regions.csv', emit: negatives
    path 'missing_accessions.csv', optional: true
    path 'invalid_positive_sites.csv', optional: true

    script:
    """
    python src/data/build_negative_sites.py \
      --positive-csv ${valid_positive} \
      --fasta ${human_fasta} \
      --output negative_phosphosites_with_regions.csv \
      --missing-output missing_accessions.csv \
      --invalid-output invalid_positive_sites.csv \
      --accession-col input_uniprot \
      --position-col input_position \
      --window 10
    """
}

process BUILD_TRAINING_DATASET {
    tag 'training dataset'

    input:
    path valid_positive, name: 'functional_score_with_regions_valid.csv'
    path negatives, name: 'negative_phosphosites_with_regions.csv'

    output:
    path 'phosphosite_training_dataset.csv', emit: training_dataset
    path 'phosphosite_excluded_rows.csv', emit: excluded_rows

    script:
    """
    python src/data/build_training_dataset.py
    """
}

process TRAIN_ONEHOT_SVM {
    tag 'one-hot SVM'

    input:
    path training_dataset, name: 'phosphosite_training_dataset.csv'

    output:
    path 'svm_outputs', emit: onehot_results
    path 'mlruns', optional: true

    script:
    """
    python src/models/train_onehot_svm.py
    """
}

process GENERATE_ESM2_EMBEDDINGS {
    tag 'ESM-2 embeddings'

    input:
    path training_dataset, name: 'phosphosite_training_dataset.csv'

    output:
    path 'plm_embeddings', emit: embeddings

    script:
    """
    python src/features/generate_esm2_embeddings.py
    """
}

process TRAIN_PLM_SVM {
    tag 'PLM SVM'

    input:
    path embeddings, name: 'plm_embeddings'

    output:
    path 'plm_svm_outputs', emit: plm_results
    path 'mlruns', optional: true

    script:
    """
    python src/models/train_plm_svm.py
    """
}

workflow {
    positive_ch = Channel.fromPath(params.positive_xlsx, checkIfExists: true)
    fasta_ch    = Channel.fromPath(params.human_fasta, checkIfExists: true)

    EXTRACT_POSITIVE_WINDOWS(positive_ch)
    FILTER_POSITIVE_SITES(EXTRACT_POSITIVE_WINDOWS.out.all_windows)
    BUILD_NEGATIVE_SITES(FILTER_POSITIVE_SITES.out.valid_positive, fasta_ch)
    BUILD_TRAINING_DATASET(
        FILTER_POSITIVE_SITES.out.valid_positive,
        BUILD_NEGATIVE_SITES.out.negatives
    )

    TRAIN_ONEHOT_SVM(BUILD_TRAINING_DATASET.out.training_dataset)
    GENERATE_ESM2_EMBEDDINGS(BUILD_TRAINING_DATASET.out.training_dataset)
    TRAIN_PLM_SVM(GENERATE_ESM2_EMBEDDINGS.out.embeddings)
}
