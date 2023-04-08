E-COMMERCE Probing tasks
============
Authors: 

Mehul Gulati (2020UCM2362) 

Jivanshu Popli (2020UCM2319) 

Vansh Kapila (2020UCM2337)

---

 The project was created during the Natural Language Processing (NLP) course at Netaji Subhas University of Technology during our 6th Semester.

---

## About

 Probing is a tool for investigating embedded spaces created using BERT-like transformers. They involve building new classifiers trained on embeddings to see if we can verify the hypothesis about the embedded space by acquiring good accuracy of the probing classifier on the embeddings.

---

## Related works

### Probing tasks
If you want to learn more about probing tasks, please read the following papers:
 - general introduction of probing tasks [Sahin et al. 2020]
 - probing tasks for visual-semantic case, this paper was our inspiration that probing tasks can be more intricate [Lindstrom et al. 2020]

### WDC dataset for e-commerce product matching
[Mozdzonek et al.2022] describes the problem and uses state-of-the-art cross-encoder to solve the problem

--- 

## Libraries
- [Hugging Face](https://huggingface.co/) [Wolf et al.2019]: A great library with many pretrained language models, such as: `bert-base-cased` or `xlm-roberta-base`.
- [SentenceTransformers](https://www.sbert.net/) [Reimers et al.2019]: A library designed for easy computing of sentence embeddings using Hugging Face transformers under the hood.
It also provides ready-to-use functions for their fine-tunning.
- [SentEval](https://github.com/facebookresearch/SentEval) [Conneau et al.2018]: SentEval is a library for evaluating the quality of sentence embeddings. They include a suite of 10 probing tasks which evaluate what linguistic properties are encoded in sentence embeddings.

---

## Datasets
- **Web Data Commons - Training Dataset and Gold Standard for Large-Scale Product Matching** - the dataset consists of pairs of offers grouped into four categories: `Computers`, `Cameras`, `Watches`, `Shoes`. Each pair of offers is either a `positive` pair (both offers regard the same product) or a `negative` pair (two different products). Please, note that offers within negative pairs regard different products but still, of the same product category (e.g. two different cameras) only.

To learn more about the dataset visit the website: [WDC dataset](http://webdatacommons.org/largescaleproductcorpus/).

---

## Project scopes

 Project covers:
 - fine-tuning of the `bert-base-cased` model on the `Cameras medium` WDC dataset (e-commerce product matching problem) 
 - calculating embeddings for each offer using the fine-tuned model
 - creating probing tasks to explain the obtained embedded space:
    - **Common words**: the goal was to build a classifier, that based on the embedding of an offer, tried to predict whether the offer contains at least one of the common words
    ('camera', 'digital', 'lens');
    - **Brand name**: the goal was to build a classifier, that based on the embedding of an offer, tried to predict whether the offer contains a brand name (e.g., 'Canon') or not;
    - **Length of sentences**: the goal was to build a classifier, that based on the embedding of an offer, tried to predict the length of the input (string containing the offer);
    - **The Levenshtein distance**: the goal was to build a classifier, that based on the pair of embeddings, tried to predict the Levenshtein distance score calculated from the two strings (each representing one offer from the pair). We discretized the target variable into 5 classes ('similar', 'slightly similar', 'neutral', 'hardly similar', 'not similar').

---

## Usage

### Notebooks
You may want to take a look at our notebooks. You can find there our work.

Project #1 notebooks:
- `finetuning_embedding_extraction.ipynb`: this notebook is meant to be run on Google Colab environment. It was used for fine-tuning of the `bert-base-cased`  model, embedding extraction and saving results.
- `probing_tasks.ipynb`: in this notebook you can find our probing tasks
- `export_tsv.ipynb`: notebook used for converting the embeddings into a .tsv file for visualization of the embedded space at https://projector.tensorflow.org/.

### Source code
All functions/classes used throughout the notebooks can be found in the `source` directory to provide easy reusing of the code in future notebooks/scripts.

### Models
To ensure reproducibility of experiments and results, we added fine-tuned models in the directories `models`.
Each model is placed in its own subdirectory (e.g., `models/bert-base-cased`) accompanied by the `info.txt` file, in which we provided additional information about fine-tuning (`hyperparameters`, used functions etc., exact dataset). The model binaries and additional configuration files (produced by the SentenceTransformers library - refer to doc) were packed into a zip archive. They are available via the url provided in  the `model_url.txt` file (due to the GitHub's limit for uploading large files).

### Scripts
Scripts are located in the directory: `scripts`

1. `train.py` - script that can be used for transformer fine-tunning: 
`python train.py --outdir ./output --hugging_face_model  bert-base-cased --batch_size 16 --dataset_type cameras --dataset_size medium --num_epochs 200`

2. `extract_embeddings.py` - extracts embeddings for a given dataset using a given model:
`python extract_embeddings.py --outdir ./output --model_inputdir ./models/bert-base-cased --dataset_type cameras --dataset_size medium`

3. `probe.py` - performs probing tasks using given embeddings and the corresponding dataset: 
`python probe.py --outdir ./output --model_inputdir ./models/bert-base-cased --dataset_type cameras --dataset_size medium`

You can use the `--help` command to learn about possible parameters passed to the scripts.

### Outputs

In the directories `project1_output`, we store all output files (such as embeddings, plots, images etc.)

### Dataset

The directory `datasets` contains data used in the project. In addition, in the directory new embeddings using for training probing tasks are saved.

---
