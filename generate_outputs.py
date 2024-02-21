import pandas as pd
from tqdm import tqdm
from transformers import pipeline

# Replace with arguments from command line.
MODEL = 'gpt2' # or 'dialogpt' or 'blenderbot'

# Models.
GPT2_MODEL = 'gpt2'
DIALOGPT_MODEL = 'dialogpt'
BLENDERBOT_MODEL = 'blenderbot'

# Holistic bias sentence indexes.
TEXT_INDEX = 0
AXIS_INDEX = 1
DESCRIPTOR_INDEX = 3
DESCRIPTOR_GENDER_INDEX = 4
NOUN_PHRASE_TYPE_INDEX = 11
TEMPLATE_INDEX = 12

def load_inputs():
  """
  Loads input prompts from dataset.
  """
  # Load dataset.
  data = pd.read_csv('/home/brian/github/generative-text-model-fairness/holistic_bias/dataset/v1.0-reduced/sentences.csv')

  # Convert to numpy array.
  sentences = data.to_numpy()

  # Filter out sentences that don't have a descriptor.
  filtered_sentences = [
    sentence
    for sentence in sentences
    if (
      sentence[NOUN_PHRASE_TYPE_INDEX]
      in [
        'descriptor_noun',
        'noun_descriptor',
      ]
      and sentence[DESCRIPTOR_GENDER_INDEX] == '(none)'
    )
  ]
  print("Filtered out", len(sentences) - len(filtered_sentences), "sentences with no descriptor.")
  print("Number of input prompts: ", len(filtered_sentences) - 1)
  return filtered_sentences

def get_output(model, text):
  """
  Gets output from specified model using input prompt.
  Model can be one of 'gpt2', 'dialogpt' or 'blenderbot'.
  """
  if model == 'gpt2':
    return gpt2_generator(text)[0]["generated_text"].replace(',', '')
  elif model == 'dialogpt':
    return None
  elif model == 'blenderbot':
    return None
  raise ValueError("Unsupported model: ", model)

if __name__ == '__main__':
  # Load model.
  if MODEL == 'gpt2':
    gpt2_generator = pipeline('text-generation', model='gpt2-large', max_new_tokens=10)
  elif MODEL == 'dialogpt':
    dialogpt_generator = pipeline(model='microsoft/DialoGPT-medium')
  elif MODEL == 'blenderbot':
    blenderbot_generator = pipeline(model='facebook/blenderbot_small-90M')

  # Load input prompts.
  inputs = load_inputs()

  # Generate outputs for each input prompt.
  with open('/home/brian/github/generative-text-model-fairness/results/'+MODEL+'-outputs.csv', 'w') as f:
    f.write(f'{"axis"},{"template"},{"descriptor"},{"input"},{"output"}\n')
    for input in tqdm(inputs):
      text = input[TEXT_INDEX]
      output = get_output(MODEL, text)
      output = " ".join(output.split())
      f.write(f'{input[AXIS_INDEX]},{input[TEMPLATE_INDEX]},{input[DESCRIPTOR_INDEX]},{text},{output}\n')
      