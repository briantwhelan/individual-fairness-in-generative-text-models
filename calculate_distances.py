import time
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from itertools import combinations
from evaluate import load
from transformers import pipeline

# Model to use for calculating distances.
MODEL = 'gpt2' # or 'blenderbot'

def load_input_output_pairs():
  """
  Loads input-output pairs from file.
  """
   # Read in input-output pairs.
  df = pd.read_csv(f'./results/{MODEL}/{MODEL}-outputs.csv')
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
  model_id = 'gpt2' if MODEL == 'gpt2' else 'facebook/blenderbot-3B'
  input_ppls = perplexity.compute(predictions=input_text, model_id=model_id)["perplexities"]
  output_ppls = perplexity.compute(predictions=output_text, model_id=model_id)["perplexities"]

  # Ensure lengths of arrays are the same.
  if (len(input_text) != len(output_text) and
      len(input_text) != len(input_ppls) and
      len(output_text) != len(output_ppls)):
    raise ValueError('Unequal array lengths: inputs (', len(input_text),
                    '), outputs (', len(output_text), '), input_ppls (', len(input_ppls),
                    '), output_ppls (', len(output_ppls), ')')
  
  # Append input and output perplexities to file.
  df = pd.read_csv(f'./results/{MODEL}/{MODEL}-outputs.csv')
  df['input_ppl'] = input_ppls
  df['output_ppl'] = output_ppls
  df.to_csv(f'./results/{MODEL}/{MODEL}-perplexities.csv', index=False)

def calculate_median_perplexities():
  """
  Calculates median perplexities of input and output texts.
  """
  # Perplexity dict indexed by template and descriptor.
  perplexities = defaultdict(lambda: defaultdict(list))
  entries = pd.read_csv(f'./results/{MODEL}/{MODEL}-perplexities.csv').to_numpy()
  for entry in entries:
    perplexities[entry[1]][entry[2]].append((entry[5], entry[6]))

  # Median perplexities per template and descriptor.
  median_perplexities = defaultdict(lambda: defaultdict(list))
  for template in perplexities:
    for descriptor in perplexities[template]:
      input_ppls, output_ppls = zip(*perplexities[template][descriptor])
      median_perplexities[template][descriptor] = (np.median(input_ppls), np.median(output_ppls))

  # Print median perplexities.
  with open(f'./results/{MODEL}/{MODEL}-median-perplexities.csv', 'w') as f:
    f.write(f'{"template"},{"descriptor"},{"median_input_ppl"},{"median_output_ppl"}\n')
    for template in median_perplexities:
      for descriptor in median_perplexities[template]:
        input_ppl, output_ppl = median_perplexities[template][descriptor]
        f.write(f'{template},{descriptor},{input_ppl},{output_ppl}\n')
  return median_perplexities

def calculate_perplexity_distances(median_perplexities):
  with open(f'./results/{MODEL}/{MODEL}-perplexity-distances.csv', 'w') as f:
    f.write(f'{"template"},{"descriptor1"},{"descriptor2"},{"input_distance"},{"output_distance"}\n')
    
    for template in median_perplexities:
      for descriptor1, descriptor2 in combinations(median_perplexities[template], r=2):
        input_distance = abs(median_perplexities[template][descriptor1][0] - median_perplexities[template][descriptor2][0])
        output_distance = abs(median_perplexities[template][descriptor1][1] - median_perplexities[template][descriptor2][1])
        f.write(f'{template},{descriptor1},{descriptor2},{input_distance},{output_distance}\n')

def calculate_sentiments(input_text, output_text):
  """
  Calculates sentiments of input and output texts.
  """
  sentiment_analysis = pipeline(model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", return_all_scores=True)

  # Calculate sentiments of inputs and outputs.
  start = time.time()
  sentiments = sentiment_analysis(input_text.tolist())
  print("Time taken to calculate input sentiments: ", time.time() - start)
  input_sentiments = []
  for sentiment in sentiments:
    input_sentiments.append((sentiment[0]['score'], sentiment[1]['score'], sentiment[2]['score']))

  start = time.time()
  sentiments = sentiment_analysis(output_text.tolist())
  print("Time taken to calculate output sentiments: ", time.time() - start)
  output_sentiments = []
  for sentiment in sentiments:
    output_sentiments.append((sentiment[0]['score'], sentiment[1]['score'], sentiment[2]['score']))

  # Ensure lengths of arrays are the same.
  if (len(input_text) != len(output_text) and
      len(input_text) != len(input_sentiments) and
      len(output_text) != len(output_sentiments)):
    raise ValueError('Unequal array lengths: inputs (', len(input_text),
                    '), outputs (', len(output_text),
                    '), input sentiments (', len(input_sentiments),
                    '), output_sentiments (', len(output_sentiments), ')')

  # Append input and output sentiments to file.
  df = pd.read_csv(f'./results/{MODEL}/{MODEL}-outputs.csv')
  pos, neu, neg = zip(*input_sentiments)
  df['input_sentiment_pos'] = pos
  df['input_sentiment_neu'] = neu
  df['input_sentiment_neg'] = neg
  pos, neu, neg = zip(*output_sentiments)
  df['output_sentiment_pos'] = pos
  df['output_sentiment_neu'] = neu
  df['output_sentiment_neg'] = neg
  df.to_csv(f'./results/{MODEL}/{MODEL}-sentiments.csv', index=False)

def calculate_median_sentiments():
  """
  Calculates median sentiments of input and output texts.
  """
  # Sentiment dict indexed by template and descriptor
  sentiments = defaultdict((lambda: defaultdict(list)))
  entries = pd.read_csv(f'./results/{MODEL}/{MODEL}-sentiments.csv').to_numpy()
  for entry in entries:
    # The positive sentiment is used here arbitrarily.
    # The neutral and negative sentiments are also valid approaches.
    sentiments[entry[1]][entry[2]].append((entry[5], entry[8]))

  # Median sentiments per template and descriptor.
  median_sentiments = defaultdict(lambda: defaultdict(list))
  for template in sentiments:
    for descriptor in sentiments[template]:
      input_sentiments, output_sentiments = zip(*sentiments[template][descriptor])
      median_sentiments[template][descriptor] = (np.median(input_sentiments), np.median(output_sentiments))

  # Print median sentiments.
  with open(f'./results/{MODEL}/{MODEL}-median-sentiments.csv', 'w') as f:
    f.write(f'{"template"},{"descriptor"},{"median_input_sentiment"},{"median_output_sentiment"}\n')
    for template in median_sentiments:
      for descriptor in median_sentiments[template]:
        input_sentiment, output_sentiment = median_sentiments[template][descriptor]
        f.write(f'{template},{descriptor},{input_sentiment},{output_sentiment}\n')
  return median_sentiments

def calculate_sentiment_distances(median_sentiments):
  with open(f'./results/{MODEL}/{MODEL}-sentiment-distances.csv', 'w') as f:
    f.write(f'{"template"},{"descriptor1"},{"descriptor2"},{"input_distance"},{"output_distance"}\n')
    for template in median_sentiments:
      for descriptor1, descriptor2 in combinations(median_sentiments[template], r=2):
        input_distance = abs(median_sentiments[template][descriptor1][0] - median_sentiments[template][descriptor2][0])
        output_distance = abs(median_sentiments[template][descriptor1][1] - median_sentiments[template][descriptor2][1])
        f.write(f'{template},{descriptor1},{descriptor2},{input_distance},{output_distance}\n')

if __name__ == '__main__':
    input_text, output_text = load_input_output_pairs()
    calculate_perplexities(input_text, output_text)
    median_perplexities = calculate_median_perplexities()
    calculate_perplexity_distances(median_perplexities)
    calculate_sentiments(input_text, output_text)
    median_sentiments = calculate_median_sentiments()
    calculate_sentiment_distances(median_sentiments)
