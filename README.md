# Entity Linking for real-time geolocation of natural disasters from tweets

This repository contains all the code and trained models required to operate the Entity Linking pipeline described in the paper written by Caillaut et al. entitled "Entity Linking for real-time geolocation of natural disasters from social network posts". This work aims to link French named entities extracted from tweet to their related spatial entities in Wikidata. 

# Repository content

1. **camembert_dual_encoder** - The [`camembert_dual_encoder`](camembert_dual_encoder/) directory contains the code required for the implementation of the model and some helpers useful for the training.

2. **Notebook:** - The [`pipeline.ipynb`](pipeline.ipynb) is a Jupyter notebook that provide step-by-step use of the pipeline, in a interactive way. 

3. **Requirements:** 
- [`environment.yml`](environment.yml) - This file lists the required dependencies for the project. These dependencies can be easily installed using a package manager like `mamba`, `conda` or `pip`.

4. **Model Scripts:** [`model.py`](model.py) - This script contains several function defining the pipeline as a processing chain involving  a _bi-encoder_ and a _cross-encoder_. The following treatments are carried out:
- The _bi-encoder_ is responsible for detecting mentions of entities and calculating their representations (_i.e. _embeddings_)
- The _embeddings_ of mentions are compared with the _embeddings_ of entities and the `k` most probable are designated as candidates.
- the _cross-encoder_ is responsible for assigning scores to candidate entities

The pipeline receives a list of character strings as input and returns a list of dictionaries containing the pipeline's predictions. Dictionaries are structured as follows:

```python
{
    'text': 'la phrase à analyser', # The sentence to analyze
    'predictions': [ # a list containing the predictions associated with the current sentence
        {
            'mention': 'le texte de la mention', # the text of the entity mention
            'start': 0, # the index of the first character of the mention
            'end': 4, # the index corresponding to the character following the last character of the statement (index of the last character + 1)
            'label': 'type de la mention', # for example GEOLOC
            'candidates': ['une', 'liste', 'd’entités', 'candidates'],
            'scores': [0.5, 1, 0.02, 0.3], # scores of candidate entities
            'wikidata': ['Q1', 'Q2', 'Q3', 'Q4'], # wikidata IDs of candidate entities
            'wikidata_properties': [ # wikidata properties extracted for each candidate entities. Values can be zero 
                {
                    'label': 'Label wikidata',
                    'description': 'Description wikidata',
                    'osm': '147559', # OpnStreetMap ID
                    'geonames': '2989317', # Geonames ID
                    'longitude': 1.9041666666667,
                    'latitude': 47.902222222222,
                    'altitude': nan,
                    'wikidata': 'Q6548' # Wikidata ID
                }
            ]
        }
    ]
}
```

# Using the Entity Linking pipeline

## Setting dependencies

Install the necessary dependencies by running the following command in your terminal:

```
mamba env create -f environment.yml # 

#Or if mamba is not available in the system
conda env create -f environment.yml

conda activate el-pipeline
```
If it is the first time your are using this code, run the `get_models_from_git.sh` script to download models required fot the Entity Linking pipeline.

## Predicting geolocation of entities
To make predictions, use the _pipeline.ipynb_ notebook.