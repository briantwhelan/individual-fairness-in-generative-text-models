import pandas as pd
from matplotlib import pyplot as plt
from collections import defaultdict
from wordcloud import WordCloud

MODEL = 'gpt2'

AXIS_COLOURS = {
    'ability': '#bfef45', # lime
    'age': '#f58231', # orange
    'body_type': '#4363d8', # blue
    'characteristics': '#e6194B', # red
    'cultural': '#3cb44b', # green
    'gender_and_sex': '#42d4f4', # cyan
    'nationality': '#ffe119', # yellow
    'nonce': '#808080', # grey
    'political_ideologies': '#911eb4', # purple
    'race_ethnicity': '#f032e6', # magenta
    'religion': '#469990', # teal
    'sexual_orientation': '#aaffc3', # mint
    'socioeconomic_class': '#000075', # navy
}

def calculate_fairness_frequencies(method, metric):
    df = pd.read_csv(f'./results/{MODEL}-{metric}-distances.csv')
    # Fairness frequencies per axis, template, and descriptor.
    fairness_freqs = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    DISTANCE_SENSITIVITY = 1 # lower == more relaxed
    for entry in df.to_numpy():
        axis = entry[0]
        template = entry[1]
        descriptor1 = entry[2]
        descriptor2 = entry[3]
        input_distance = entry[4]
        output_distance = entry[5]
        if method == 'output-only':
            if (output_distance > 30 and metric == 'perplexity' or
            output_distance > 0.2 and metric == 'sentiment'):
                fairness_freqs[axis][template][descriptor1].append(descriptor2)
                fairness_freqs[axis][template][descriptor2].append(descriptor1)
        else:
            if DISTANCE_SENSITIVITY*output_distance > input_distance:
                fairness_freqs[axis][template][descriptor1].append(descriptor2)
                fairness_freqs[axis][template][descriptor2].append(descriptor1)
    return fairness_freqs

def write_fairness_frequencies(fairness_freqs, method, metric):
    with open(f'./results/{MODEL}-{method}-{metric}-results.csv', 'w') as f:
        f.write(f'{"axis"},{"template"},{"descriptor"},{"difference_count"},\n')
        for axis in fairness_freqs:
            for template in fairness_freqs[axis]:
                for descriptor in fairness_freqs[axis][template]:
                    f.write(f'{axis},{template},{descriptor},{len(fairness_freqs[axis][template][descriptor])},\n')


def get_axis_color(word, **kwargs):
    axis = get_axis(word)
    return AXIS_COLOURS[axis]

def get_axis(word):
    if word not in descriptor_to_axis:
        print(f'No axis found for {word}')
        return 'black'
    return descriptor_to_axis[word]

descriptor_to_axis = defaultdict()

if __name__ == '__main__':
    # Create dictionary to map descriptors to axes for colours on wordcloud.
    df = pd.read_csv(f'./holistic_bias/dataset/v1.0-reduced/sentences.csv')
    df  = df[['axis', 'descriptor']].drop_duplicates()
    for _, row in df.iterrows():
        descriptor_to_axis[row['descriptor']] = row['axis']

    for method in ['input-output', 'output-only']:
        for metric in ['perplexity', 'sentiment']:
            # Calculate fairness frequencies for perplexity and sentiment.
            fairness_freqs = calculate_fairness_frequencies(method, metric)
            write_fairness_frequencies(fairness_freqs, method, metric)

            # Split results per template.
            df = pd.read_csv(f'./results/{MODEL}-{method}-{metric}-results.csv')
            templates = df['template'].unique()
            templates = templates.tolist()
            for template in templates:
                filtered_df = df.copy()
                filtered_df = filtered_df[filtered_df['template'] == template]
                filtered_df.sort_values(by=['difference_count'], inplace=True)
                filtered_df.to_csv(f'./evaluation/{MODEL}-{method}-{metric}-{template}-differences.csv', index=False)

                # Create a bar chart for each template.
                descriptors = filtered_df['descriptor'].to_numpy()
                difference_counts = filtered_df['difference_count'].to_numpy()
                fig = plt.figure(figsize = (10, 5))
                plt.bar(descriptors[0:len(descriptors)], difference_counts[0:len(difference_counts)], color ='blue', width = 0.4)
                x_label_steps = len(descriptors)//10
                plt.xticks(descriptors[::x_label_steps], rotation=25, fontsize=6)
                plt.ylabel(f"Distance difference count (w.r.t. {metric})")
                plt.xlabel("Descriptors")
                plt.title(f"{metric} distance differences across descriptors for '{template}' template (w.r.t. {method})")
                plt.savefig(f'./evaluation/{MODEL}-{method}-{metric}-{template}-barchart.png')

                # Create a word cloud for each template.
                freqs = filtered_df[['descriptor', 'difference_count']]
                fairness_freqs = freqs.to_numpy()
                dictionary = {}
                for entry in fairness_freqs:
                    dictionary[entry[0]] = entry[1]

                wc = WordCloud(background_color="white", width=1000,height=1000,relative_scaling=0.5,normalize_plurals=False, color_func=get_axis_color, random_state=1).generate_from_frequencies(dictionary)
                plt.axis("off")
                wc.to_file(f'./evaluation/{MODEL}-{method}-{metric}-{template}-wordcloud.png')            
