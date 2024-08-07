# Usage

1. Create the embeddings
2. Create the aggregated embeddings
3. Run the recognizer
4. Run postprocessing

# Create results data on exacloud:

```python3 src/post_processing/aggregate_run_metrics.py -d ./results/simple_recognizer/ -o /home/groups/EllrottLab/heterogeneous-embedding-vectors/results/aggregated_metrics/simple_recognizer```

```python3 src/post_processing/aggregate_run_metrics.py -d ./results/simple_recognizer_foundation/ -o /home/groups/EllrottLab/heterogeneous-embedding-vectors/results/aggregated_metrics/simple_recognizer_foundation```

```python3 src/post_processing/aggregate_run_metrics.py -d ./results/multi_recognizer/brca_laml -o /home/groups/EllrottLab/heterogeneous-embedding-vectors/results/aggregated_metrics/multi_recognizer/brca_laml```

```python3 src/post_processing/aggregate_run_metrics.py -d ./results/multi_recognizer_foundation/brca_laml -o /home/groups/EllrottLab/heterogeneous-embedding-vectors/results/aggregated_metrics/multi_recognizer_foundation/brca_laml```

## Download data from exacloud:
    
```scp -r kirchgae@exahead1.ohsu.edu:/home/groups/EllrottLab/heterogeneous-embedding-vectors/results/aggregated_metrics/ ./results/```
    
```scp -r kirchgae@exahead1.ohsu.edu:/home/groups/EllrottLab/heterogeneous-embedding-vectors/results/aggregated_metrics/simple_recognizer_foundation ./results/aggregated_metrics/```
    
```scp -r kirchgae@exahead1.ohsu.edu:/home/groups/EllrottLab/heterogeneous-embedding-vectors/results/aggregated_metrics/multi_recognizer/brca_laml ./results/aggregated_metrics/```
    
```scp -r kirchgae@exahead1.ohsu.edu:/home/groups/EllrottLab/heterogeneous-embedding-vectors/results/aggregated_metrics/multi_recognizer_foundation/brca_laml ./results/aggregated_metrics/```
    

## Realistic Recognizer

###
Requirements:
- tesseract


## Cancers used
(STAD) Stomach adenocarcinoma
(COAD) Colon adenocarcinoma 
(LAML) Acute Myeloid Leukemia
(BRCA) Breast invasive carcinoma
(THCA) Thyroid carcinoma
(BLCA) Bladder Urothelial Carcinoma