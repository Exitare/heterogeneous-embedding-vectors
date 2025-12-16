# Cancers used
- (STAD) Stomach adenocarcinoma  
- (COAD) Colon adenocarcinoma   
- (LAML) Acute Myeloid Leukemia 
- (BRCA) Breast invasive carcinoma
- (THCA) Thyroid carcinoma
- (BLCA) Bladder Urothelial Carcinoma

# Disclaimer

Running the recognizer requires significant computational resources.
# Usage

## Install requirements
```pip install -r requirements.txt```


## Data preparation

To prepare the data yourself, use this command to run the data preparation pipeline:  
```snakemake -s ./src/data_preparation/Snakefile --cores 1```

To use already prepared data run the data_download script:  
```python src/data_download.py```  

It will download the data from a public repository to the appropriate folders.

## Running the holdout classifier 

### Summed Embeddings
```snakemake -s ./src/classifier_holdout/summed_embeddings_generation/Snakefile --cores 1```

### Models
```snakemake -s ./src/classifier_holdout/models/Snakefile --cores 1```

### Plots
```snakemake -s ./src/classifier_holdout/plots/Snakefile --cores 1```


## Latent Dimensions Classifier

### Summed Embeddings
```snakemake -s ./src/classifier_latent_dims/summed_embeddings_generation/Snakefile --cores 1```
### Models
```snakemake -s ./src/classifier_latent_dims/models/Snakefile --cores 1```
### Plots
```snakemake -s ./src/classifier_latent_dims/plots/Snakefile --cores 1```


## Subtype Classifier
### Summed Embeddings
```snakemake -s ./src/sub_type_classifier/summed_embeddings_generation/Snakefile --cores 1```

### Models
```snakemake -s ./src/sub_type_classifier/models/Snakefile --cores 1```

### Plots
```snakemake -s ./src/sub_type_classifier/plots/Snakefile --cores 1```


# TMB Classifier
### Summed Embeddings
```snakemake -s ./src/tmb_classifier/summed_embeddings_generation/Snakefile --cores 1```

### Models
```snakemake -s ./src/tmb_classifier/models/Snakefile --cores 1```

### Plots
```snakemake -s ./src/tmb_classifier/plots/Snakefile --cores 1```

## Recognizer

### Summed Embeddings
```snakemake -s ./src/recognizer/summed_embeddings_generation/Snakefile --cores 1```

### Models
```snakemake -s ./src/recognizer/models/Snakefile --cores 1```

### Plots
```snakemake -s ./src/recognizer/plots/Snakefile --cores 1```

