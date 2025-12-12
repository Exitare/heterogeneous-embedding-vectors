# Cancers used
- (STAD) Stomach adenocarcinoma  
- (COAD) Colon adenocarcinoma   
- (LAML) Acute Myeloid Leukemia 
- (BRCA) Breast invasive carcinoma
- (THCA) Thyroid carcinoma
- (BLCA) Bladder Urothelial Carcinoma


# Usage

## Install requirements
```pip install -r requirements.txt```


## Data preparation
```snakemake -s ./src/data_preparation/Snakefile --cores 1```

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

