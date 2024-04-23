# Individual Fairness in Generative Text Models
This repository contains the data and code used in completion of the dissertation 'Individual Fairness in Generative Text Models'.

# Abstract
Generative text models, specifically Large Language Models (LLMs), have bridged the communication divide between humans and computers by enabling people to use natural language to interact with state-of-the-art models. This advancement raises concerns about the extent to which such models perpetuate biases and prejudices through their responses. However, in spite of this concern, research in bias and fairness for generative text models remains sparse. Hence, this dissertation aims to add to the existing literature by proposing a method for algorithmically evaluating the individual fairness of generative text models,
where individual fairness captures the notion that similar individuals should be treated
similarly.

This work formally defines the notion of an ‘individual’ within generative text model fairness and quantifies the similarity across individuals through the definition of a similarity
metric. A fairness criterion is then defined which encodes the notion of individual fairness
by specifying that the distance between the responses for a given generative text model
given some input prompts, should be no greater than the distance between the respective
input prompts, where the distance is quantified using the similarity metric. This fairness
criterion is incorporated into existing dataset-based methods for identifying biases in NLP
models to propose, to the best of the author’s knowledge, the first method for assessing
individual fairness in a generative text model.
Evaluating this method against two state-of-the-art generative text models with known
biases using two different similarity metrics, the results offer positive evidence for incorporating additional context, through a similarity metric, into methods for evaluating
individual fairness in generative text models. While the method proposed is not without limitations, it can hopefully serve as initial motivation for future work in individual
fairness in generative text models.

# Guide to the Repository
To reproduce results within the dissertation and further future efforts, the structure of the respository along with the commands necessary to run the various scripts are documented below.

## Installation
From the root folder of the repository, install the necessary libraries by running:
```sh
$ pip install -r requirements.txt
```

## Generating the Dataset
A fork of the code for generating the [HolisticBias](https://github.com/facebookresearch/ResponsibleNLP/tree/main/holistic_bias) dataset is located at `/holistic_bias`, with the newly created `v1.0-reduced` dataset components found at `/holisticbias/dataset/v1.0-reduced`. To generate the dataset, from the root folder of the repository, run:

```sh
$ python3 ./holistic_bias/generate_sentences.py
```

The `v1.0-reduced` dataset should then be available as `sentences.csv` in  `/holisticbias/dataset/v1.0-reduced`.

## Generating Outputs
Using the `v1.0-reduced` dataset, corresponding outputs for each of the input prompts are generated using either the [`gpt2-large`](https://huggingface.co/openai-community/gpt2-large) or [`blenderbot-3B`](https://huggingface.co/facebook/blenderbot-3B) by modifying the constant `MODEL` at the top of the `generate_outputs.py` script. Run the script using the following command from the root folder of the repository:

```sh
$ python3 generate_outputs.py
```

The model outputs should be available as `<MODEL>-outputs.csv` in `/results/<MODEL>`.

## Calculating Distances
For the selected `MODEL`, which should be specified at the top of the `calculate_distances.py` script, distances between inputs and their respective outputs are calculated using both perplexity and sentiment analysis similarity metrics to quantify the distance between texts. To calculate the distances, run:

```sh
$ python3 calculate_distances.py
```

The perplexity and sentiment analysis results should be available as `<MODEL>-perplexities.csv` and `<MODEL>-sentiments.csv` for each of the respective metrics in `/results/<MODEL>`. The median results across all template-descriptor pairs for each of the respective metrics should also be available as `<MODEL>-median-perplexities.csv` and `<MODEL>-median-sentiments.csv` in `/results/<MODEL>` also. Finally, the distances should be available as `<MODEL>-perplexity-distances.csv` and `<MODEL>-sentiment-distances.csv` for each of the respective metrics in `/results/<MODEL>`. 

## Creating Plots
Using the distances and the specified fairness criterion, a series of plots are created corresponding to the number of Fairness Criterion Violations (FCVs). Two fairness criterions are assessed, namely the baseline `output-only` method's fairness criterion and the fairness criterion of the `input-output` method proposed in this work. To create the plots, run:

```sh
$ python3 create_plots.py
```

The results of the numbers of FCVs under each of the respective method's fairness criterions and each of the similarity metrics should be available as `<MODEL>-<METHOD>-<METRIC>-results.csv` in `/results/<MODEL>`. The results, split by template, are also available as `<MODEL>-<METHOD>-<METRIC>-<TEMPLATE>-differences.csv` in `/evaluation/<MODEL>/differences/<METHOD>`. Barcharts of these results, grouped by demographic axis, are available as `<MODEL>-<METHOD>-<METRIC>-<TEMPLATE>-barchart.png` in `/evaluation/<MODEL>/barcharts/<METHOD>`. Finally, wordclouds, coloured by demographic axis, are available as  `<MODEL>-<METHOD>-<METRIC>-<TEMPLATE>-wordcloud.png` in `/evaluation/<MODEL>/wordclouds/<METHOD>`.

## Putting It All Together
Once the dataset has been created, the remaining tasks, i.e. output generation, distance calculation and plot creation, can all be done using the `run.sh` script* by running:

```sh
$ ./run.sh
```

*Note that the script still requires that the `MODEL` constant in each of the respective scripts is set correctly corresponding to the model under consideration.


# Future Work
Future work should look at refining this method with bigger datasets (e.g. more templates, descriptors, nouns), more metrics (e.g. [BLEURT](https://github.com/google-research/bleurt)) and most importantly, more models (e.g. [Gemini](https://gemini.google.com/), [GPT-4](https://openai.com/gpt-4)).

If you have any questions, please file an issue in this repository and contributions are welcomed.

More than anything, it is hoped that this work will inspire future work in the area of fairness in generative text models.
