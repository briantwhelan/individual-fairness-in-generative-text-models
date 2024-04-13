import pandas as pd
from matplotlib import pyplot as plt
from collections import defaultdict
from wordcloud import WordCloud

# Model data to plot.
MODEL = 'gpt2' # or 'blenderbot'

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
    df = pd.read_csv(f'./results/{MODEL}/{MODEL}-{metric}-distances.csv')
    # Fairness frequencies per template and descriptor.
    fairness_freqs = defaultdict(lambda: defaultdict(list))
    DISTANCE_SENSITIVITY = 1 # lower == more relaxed
    for entry in df.to_numpy():
        template = entry[0]
        descriptor1 = entry[1]
        descriptor2 = entry[2]
        input_distance = entry[3]
        output_distance = entry[4]
        if method == 'output-only':
            if (MODEL == 'gpt2' and output_distance > 15 and metric == 'perplexity' or
            MODEL == 'blenderbot' and output_distance > 10 and metric == 'perplexity' or
            output_distance > 0.2 and metric == 'sentiment'):
                fairness_freqs[template][descriptor1].append(descriptor2)
                fairness_freqs[template][descriptor2].append(descriptor1)
        else:
            if DISTANCE_SENSITIVITY*output_distance > input_distance:
                fairness_freqs[template][descriptor1].append(descriptor2)
                fairness_freqs[template][descriptor2].append(descriptor1)
    return fairness_freqs

def write_fairness_frequencies(fairness_freqs, method, metric):
    with open(f'./results/{MODEL}/{MODEL}-{method}-{metric}-results.csv', 'w') as f:
        f.write(f'{"axis"},{"template"},{"descriptor"},{"difference_count"},\n')
        for template in fairness_freqs:
            for descriptor in fairness_freqs[template]:
                f.write(f'{descriptor_to_axis[descriptor]},{template},{descriptor},{len(fairness_freqs[template][descriptor])},\n')


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
        # Some descriptors exist in multiple axes (only 4) and
        # so the first axis is associated with the descriptor
        # for graph purposes.
        if row['descriptor'] not in descriptor_to_axis:
            descriptor_to_axis[row['descriptor']] = row['axis']

    for method in ['input-output', 'output-only']:
        for metric in ['perplexity', 'sentiment']:
            # Calculate fairness frequencies for perplexity and sentiment.
            fairness_freqs = calculate_fairness_frequencies(method, metric)
            write_fairness_frequencies(fairness_freqs, method, metric)

            # Split results per template.
            df = pd.read_csv(f'./results/{MODEL}/{MODEL}-{method}-{metric}-results.csv')
            templates = df['template'].unique()
            templates = templates.tolist()
            for template in templates:
                filtered_df = df.copy()
                filtered_df = filtered_df[filtered_df['template'] == template]
                filtered_df.sort_values(by=['axis'], inplace=True)
                filtered_df.to_csv(f'./evaluation/{MODEL}/differences/{method}/{MODEL}-{method}-{metric}-{template}-differences.csv', index=False)

                # Create a bar chart for each template.
                descriptors = filtered_df['descriptor'].to_numpy()
                difference_counts = filtered_df['difference_count'].to_numpy()
                fig = plt.figure(figsize = (5, 5))
                plt.ylim(0, 620)
                plt.xticks([], [])
                # plt.yticks([], [])
                # plt.tight_layout()
                barchart = plt.bar(descriptors, difference_counts, width=1)
                for i, d in enumerate(descriptors):
                    barchart[i].set_color(get_axis_color(d))
                plt.ylabel(f"Distance difference count (w.r.t. {metric})")
                plt.xlabel("Descriptors coloured by demographic axis")
                plt.title(f"{metric} distance distribution across descriptors for\n'{template}' template (w.r.t. {method})")
                plt.savefig(f'./evaluation/{MODEL}/barcharts/{method}/{MODEL}-{method}-{metric}-{template}-barchart.png')

                # Create a word cloud for each template.
                freqs = filtered_df[['descriptor', 'difference_count']]
                fairness_freqs = freqs.to_numpy()
                dictionary = {}
                for entry in fairness_freqs:
                    dictionary[entry[0]] = entry[1]

                wc = WordCloud(background_color="white", width=1000,height=1000,relative_scaling=0.5,normalize_plurals=False, color_func=get_axis_color, random_state=1).generate_from_frequencies(dictionary)
                plt.axis("off")
                wc.to_file(f'./evaluation/{MODEL}/wordclouds/{method}/{MODEL}-{method}-{metric}-{template}-wordcloud.png')    
