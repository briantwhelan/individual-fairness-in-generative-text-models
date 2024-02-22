import pandas as pd
from matplotlib import pyplot as plt
from collections import defaultdict
from wordcloud import WordCloud

MODEL = 'gpt2'

def calculate_fairness_frequencies(metric):
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
        if DISTANCE_SENSITIVITY*output_distance > input_distance:
            fairness_freqs[axis][template][descriptor1].append(descriptor2)
            fairness_freqs[axis][template][descriptor2].append(descriptor1)
    return fairness_freqs

def write_fairness_frequencies(fairness_freqs, metric):
    with open(f'./results/{MODEL}-{metric}-results.csv', 'w') as f:
        f.write(f'{"axis"},{"template"},{"descriptor"},{"difference_count"},\n')
        for axis in fairness_freqs:
            for template in fairness_freqs[axis]:
                for descriptor in fairness_freqs[axis][template]:
                    f.write(f'{axis},{template},{descriptor},{len(fairness_freqs[axis][template][descriptor])},\n')

if __name__ == '__main__':
    for metric in ['perplexity', 'sentiment']:
        # Calculate fairness frequencies for perplexity and sentiment.
        fairness_freqs = calculate_fairness_frequencies('perplexity')
        write_fairness_frequencies(fairness_freqs, 'perplexity')

        fairness_freqs = calculate_fairness_frequencies('sentiment')
        write_fairness_frequencies(fairness_freqs, 'sentiment')

        # Split results per template.
        df = pd.read_csv(f'./results/{MODEL}-{metric}-results.csv')
        templates = df['template'].unique()
        templates = templates.tolist()
        for template in templates:
            df = pd.read_csv(f'./results/{MODEL}-{metric}-results.csv')
            df = df[df['template'] == template]
            df.sort_values(by=['difference_count'], inplace=True)
            df.to_csv(f'./evaluation/{MODEL}-{template}-{metric}-differences.csv', index=False)

            # Create a bar chart for each template.
            descriptors = df['descriptor'].to_numpy()
            difference_counts = df['difference_count'].to_numpy()
            fig = plt.figure(figsize = (10, 5))
            plt.bar(descriptors[0:len(descriptors)], difference_counts[0:len(difference_counts)], color ='blue', width = 0.4)
            x_label_steps = len(descriptors)//10
            plt.xticks(descriptors[::x_label_steps], rotation=25, fontsize=6)
            plt.ylabel(f"Distance difference count (w.r.t. {metric})")
            plt.xlabel("Descriptors")
            plt.title(f"{metric} distance differences across descriptors for '{template}' template")
            plt.savefig(f'./evaluation/{MODEL}-{template}-{metric}-barchart.png')

            # Create a word cloud for each template.
            df = df[['descriptor', 'difference_count']]
            fairness_freqs = df.to_numpy()
            dictionary = {}
            for entry in fairness_freqs:
                dictionary[entry[0]] = entry[1]

            wc = WordCloud(background_color="white", width=1000,height=1000,relative_scaling=0.5,normalize_plurals=False).generate_from_frequencies(dictionary)
            plt.axis("off")
            wc.to_file(f'./evaluation/{MODEL}-{template}-{metric}-wordcloud.png')
            
