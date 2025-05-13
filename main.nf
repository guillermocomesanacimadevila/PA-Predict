#!/usr/bin/env nextflow

// --------------------
// Parameters
// --------------------
params.samples     = 1000
params.seed        = 42
params.model_type  = 'rf'
params.outdir      = "output"
params.dataset     = "${params.outdir}/simulated_pa_data.csv"
params.report_csv  = "${params.outdir}/model_comparison.csv"
params.report_html = "${params.outdir}/report.html"
params.template    = "Scripts/report_template.html"

// --------------------
// Process 1: Generate Synthetic Dataset
// --------------------
process GenerateData {
    publishDir "${params.outdir}", mode: 'copy'

    output:
    path "${params.dataset}", emit: dataset

    script:
    """
    mkdir -p ${params.outdir}
    python Scripts/generate_pa_data.py \\
      --samples ${params.samples} \\
      --seed ${params.seed} \\
      --output ${params.dataset}
    """
}

// --------------------
// Process 2: Train Model
// --------------------
process TrainModel {
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path dataset from GenerateData.dataset

    output:
    path "${params.outdir}/model.pkl"
    path("${params.outdir}/figs"), emit: figs_out

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

// --------------------
// Process 3: Benchmark All Models
// --------------------
process BenchmarkModels {
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path dataset from GenerateData.dataset

    output:
    path "${params.report_csv}", emit: benchmark_csv

    script:
    """
    python Scripts/benchmark_models.py \\
      --data ${dataset} \\
      --output_csv ${params.report_csv}
    """
}

// --------------------
// Process 4: Generate HTML Report
// --------------------
process GenerateReport {
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path benchmark_csv from BenchmarkModels.benchmark_csv

    output:
    path "${params.report_html}"

    script:
    """
    python Scripts/generate_report.py \\
      --csv ${benchmark_csv} \\
      --output ${params.report_html} \\
      --template ${params.template}
    """
}

// --------------------
// Workflow Execution Order
// --------------------
workflow {
    GenerateData()
    TrainModel(GenerateData.out)
    BenchmarkModels(GenerateData.out)
    GenerateReport(BenchmarkModels.out)
}
