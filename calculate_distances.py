import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from itertools import combinations
from evaluate import load
from transformers import pipeline

# Replace with arguments from command line.
MODEL = 'gpt2' # or 'dialogpt' or 'blenderbot'

def load_input_output_pairs():
  """
  Loads input-output pairs from file.
  """
   # Read in input-output pairs.
  df = pd.read_csv('./results/'+MODEL+'-outputs.csv')
  input_text = df['input'].to_numpy()
  output_text = df['output'].to_numpy()

  if len(input_text) != len(output_text):
    raise ValueError('Unequal input-output lengths: inputs (', len(input_text),
                    '), outputs (', len(output_text), ')')
  
  return input_text, output_text

def calculate_perplexities(input_text, output_text):
  """
  Calculates perplexities of input and output texts.
  """
  perplexity = load('perplexity', module_type='metric')

  # Calculate perplexities of inputs and outputs.
  input_ppls = perplexity.compute(predictions=input_text, model_id='gpt2')["perplexities"]
  output_ppls = perplexity.compute(predictions=output_text, model_id='gpt2')["perplexities"]

  # Ensure lengths of arrays are the same.
  if (len(input_text) != len(output_text) and
      len(input_text) != len(input_ppls) and
      len(output_text) != len(output_ppls)):
    raise ValueError('Unequal array lengths: inputs (', len(input_text),
                    '), outputs (', len(output_text), '), input_ppls (', len(input_ppls),
                    '), output_ppls (', len(output_ppls), ')')
  
  # Append input and output perplexities to file.
  df = pd.read_csv('./results/'+MODEL+'-outputs.csv')
  df['input_ppl'] = input_ppls
  df['output_ppl'] = output_ppls
  df.to_csv('./results/'+MODEL+'-perplexities.csv', index=False)

def calculate_median_perplexities():
  """
  Calculates median perplexities of input and output texts.
  """
  # Perplexity dict indexed by axis, template, and descriptor.
  perplexities = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
  entries = pd.read_csv('./results/'+MODEL+'-perplexities.csv').to_numpy()
  for entry in entries:
    perplexities[entry[0]][entry[1]][entry[2]].append((entry[5], entry[6]))

  # Median perplexities per axis, template, and descriptor.
  median_perplexities = defaultdict(
              lambda: defaultdict(lambda: defaultdict(list))
          )
  for axis in perplexities:
    for template in perplexities[axis]:
      for descriptor in perplexities[axis][template]:
        input_ppls, output_ppls = zip(*perplexities[axis][template][descriptor])
        median_perplexities[axis][template][descriptor] = (np.median(input_ppls), np.median(output_ppls))

  # Print median perplexities.
  with open('./results/'+MODEL+'-median-perplexities.csv', 'w') as f:
    f.write(f'{"axis"},{"template"},{"descriptor"},{"median_input_ppl"},{"median_output_ppl"}\n')
    for axis in median_perplexities:
      for template in median_perplexities[axis]:
        for descriptor in median_perplexities[axis][template]:
          input_ppl, output_ppl = median_perplexities[axis][template][descriptor]
          f.write(f'{axis},{template},{descriptor},{input_ppl},{output_ppl}\n')
  return median_perplexities

def calculate_perplexity_distances(median_perplexities):
  with open('./results/'+MODEL+'-perplexity-distances.csv', 'w') as f:
    f.write(f'{"axis"},{"template"},{"descriptor1"},{"descriptor2"},{"input_distance"},{"output_distance"}\n')
    
    for axis in median_perplexities:
      for template in median_perplexities[axis]:
        for descriptor1, descriptor2 in combinations(median_perplexities[axis][template], r=2):
          input_distance = abs(median_perplexities[axis][template][descriptor1][0] - median_perplexities[axis][template][descriptor2][0])
          output_distance = abs(median_perplexities[axis][template][descriptor1][1] - median_perplexities[axis][template][descriptor2][1])
          f.write(f'{axis},{template},{descriptor1},{descriptor2},{input_distance},{output_distance}\n')

def calculate_sentiments(input_text, output_text):
  """
  Calculates sentiments of input and output texts.
  """
  # sentiment_analysis = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")
  sentiment_analysis = pipeline(model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", return_all_scores=True)

  # Calculate sentiments of inputs and outputs.
  input_sentiments = []
  for input in tqdm(input_text):
    sentiment = sentiment_analysis(input)[0]
    if sentiment['label'] == 'POSITIVE':
      input_sentiments.append(sentiment['score'])
    else:
      input_sentiments.append(-sentiment['score'])
  output_sentiments = []
  for output in tqdm(output_text):
    sentiment = sentiment_analysis(output)[0]
    if sentiment['label'] == 'POSITIVE':
      output_sentiments.append(sentiment['score'])
    else:
      output_sentiments.append(-sentiment['score'])

  # Ensure lengths of arrays are the same.
  if (len(input_text) != len(output_text) and
      len(input_text) != len(input_sentiments) and
      len(output_text) != len(output_sentiments)):
    raise ValueError('Unequal array lengths: inputs (', len(input_text),
                    '), outputs (', len(output_text),
                    '), input sentiments (', len(input_sentiments),
                    '), output_sentiments (', len(output_sentiments), ')')

  # Append input and output sentiments to file.
  df = pd.read_csv('./results/'+MODEL+'-outputs.csv')
  df['input_sentiment'] = input_sentiments
  df['output_sentiment'] = output_sentiments
  df.to_csv('./results/'+MODEL+'-sentiments.csv', index=False)

def calculate_median_sentiments():
  """
  Calculates median sentiments of input and output texts.
  """
  sentiments = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
  entries = pd.read_csv('./results/'+MODEL+'-sentiments.csv').to_numpy()
  for entry in entries:
    sentiments[entry[0]][entry[1]][entry[2]].append((entry[5], entry[6]))

  # Median sentiments per axis, template, and descriptor.
  median_sentiments = defaultdict(
              lambda: defaultdict(lambda: defaultdict(list))
          )
  for axis in sentiments:
    for template in sentiments[axis]:
      for descriptor in sentiments[axis][template]:
        input_sentiments, output_sentiments = zip(*sentiments[axis][template][descriptor])
        median_sentiments[axis][template][descriptor] = (np.median(input_sentiments), np.median(output_sentiments))

  # Print median sentiments.
  with open('./results/'+MODEL+'-median-sentiments.csv', 'w') as f:
    f.write(f'{"axis"},{"template"},{"descriptor"},{"median_input_sentiment"},{"median_output_sentiment"}\n')
    for axis in median_sentiments:
      for template in median_sentiments[axis]:
        for descriptor in median_sentiments[axis][template]:
          input_sentiment, output_sentiment = median_sentiments[axis][template][descriptor]
          f.write(f'{axis},{template},{descriptor},{input_sentiment},{output_sentiment}\n')
  return median_sentiments

def calculate_sentiment_distances(median_sentiments):
  with open('./results/'+MODEL+'-sentiment-distances.csv', 'w') as f:
    f.write(f'{"axis"},{"template"},{"descriptor1"},{"descriptor2"},{"input_distance"},{"output_distance"}\n')
    
    for axis in median_sentiments:
      for template in median_sentiments[axis]:
        for descriptor1, descriptor2 in combinations(median_sentiments[axis][template], r=2):
          input_distance = abs(median_sentiments[axis][template][descriptor1][0] - median_sentiments[axis][template][descriptor2][0])
          output_distance = abs(median_sentiments[axis][template][descriptor1][1] - median_sentiments[axis][template][descriptor2][1])
          f.write(f'{axis},{template},{descriptor1},{descriptor2},{input_distance},{output_distance}\n')

if __name__ == '__main__':
    input_text, output_text = load_input_output_pairs()
    calculate_perplexities(input_text, output_text)
    median_perplexities = calculate_median_perplexities()
    calculate_perplexity_distances(median_perplexities)
    calculate_sentiments(input_text, output_text)
    median_sentiments = calculate_median_sentiments()
    calculate_sentiment_distances(median_sentiments)