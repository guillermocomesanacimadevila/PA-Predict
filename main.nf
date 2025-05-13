#!/usr/bin/env nextflow

params.samples = 1000
params.seed = 42
params.model_type = 'rf'
params.outdir = "output"
params.dataset = "${params.outdir}/simulated_pa_data.csv"

process GenerateData {
    output:
    path "output/simulated_pa_data.csv", emit: dataset

    script:
    """
    mkdir -p output
    python Scripts/generate_pa_data.py \\
      --samples ${params.samples} \\
      --seed ${params.seed} \\
      --output ${params.dataset}
    """
}

process TrainModel {
    input:
    path dataset from GenerateData.dataset

    output:
    path "${params.outdir}/model.pkl"
    path "${params.outdir}/figs/*"

    script:
    """
    mkdir -p ${params.outdir}/figs
    python Scripts/pa_model_trainer.py \\
      --data ${dataset} \\
      --model ${params.model_type} \\
      --savefigs \\
      --output_model ${params.outdir}/model.pkl \\
      --output_figs_dir ${params.outdir}/figs
    """
}
