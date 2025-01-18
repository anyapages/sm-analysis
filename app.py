import re  # For hashtag extraction
import matplotlib.pyplot as plt  # For plotting the graphs
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # For VADER sentiment analysis
import pandas as pd
import seaborn as sns
import itertools  # For co-occurrence analysis

# Initialise VADER sentiment analyser
analyzer = SentimentIntensityAnalyzer()

# Public Instagram posts
posts = [
    "If we say it's okay to not be okay we really need to back it up. ... #MentalHealthCare #Wellbeing #Anxiety #Depression #Grief #Trauma",
    "A gentle reminder. What does your nervous system need today? 🤍 #selflove #selfcare #mentalhealth #mentalwellness #youareenough",
    "Today I’m sharing 6 mental health reminders 🤍 Which one resonates with you? #mentalhealthawareness #selfcare #mentalhealth",
    "Whenever anxiety visits you, take a deep breath and think about all the great things in life. #anxiety #mentalhealthsupport",
    "Do you know who’s good for your mental health? #SelfCare #MentalHealthMatters #MentalWellness #MentalHealthSupport",
    "TBH, it’s a form of self-care 🫠 #motherhood #momlife #selfcare #coffee",
    "What are your favourite self care activities? Mine are: walking in nature, detoxing from my phone, plant care... #selfcare",
    "Remember, caring for others starts with taking care of yourself 💕 #mentalhealthawareness #mentalhealth #selfcare",
    "Nothing to see here, just some normal things that happen in therapy. 🛋️ #mentalhealthawareness #therapyworks",
    "Feed your mind with good thoughts and watch yourself grow. 🌱✨ #PositiveThinking #MentalHealthMatters #SelfCare #Mindfulness #GoodVibes #MentalWellness",
    "Unfortunately it is too expensive. What are some things that are getting too expensive for your mental health?",
    "if my fridge doesn’t stop making noises i sWEARRRRRRAHHYHGGG #anxiety #anxietyawareness #mentalhealth #mentalhealthart #arttherapy #inkdrawing",
    "Sitting in my calm era. What is will be and I will sit back and let it unfold. ... #Findingbalance #communityofquotes #mentalwellness",
    "The Trauma Brain 🧠 ✍️🏼 This is a very basic explanation of how our amygdala and hippocampus work together during trauma... #psychology #mentalhealth #trauma",
    "Who else monetizes your problems?🤣… take care of your mental health.",
    "It’s okay to put it down 🤍 Cheers, Steph x #selfcare #mentalhealth #selflove",
    "it’s okay. if you feel lost try out my app GRO. i created it for people to help them find their own light in a dark world. #mensmentalhealth",
    "When we experience anxiety, our body goes into a mode that we often call ’fight or flight’. 🍓 What stress response do you experience? 🍓 #mentalhealth #anxiety",
    "Drop three ♥♥♥ in the comments if you can relate! Here's a post by @fightthroughmentalhealth about depression... #mentalhealthawareness",
    "According to research, up to 90% of the things people worry about never happen. ... #mentalhealth #takeiteasy",
    "Sometimes it’s really hard to verbalize what we want other people to know about our struggles. So I wanted to create something that could help a little bit. Pass this along to those who might need to see it. #mentalhealth #mentalillnessawareness #therapistsofinstagram",
    "It’s a common symptom of many mental health issues like depression.",
    "The science of feeling good. 🧠 About five million Australians will experience mental illness in any given year. #MentalHealth #ImproveYourMood #CSIRO",
    "Do you ever look back at the past and see the storms that have come and the impact they have had on your life? #Findingbalance #communityofquotes #mentalwellness",
    "OCD - obsessive-compulsive disorder - is not an adjective or a personality trait. Using phrases like 'I’m so OCD' can perpetuate misconceptions. #mentalhealth #OCDawareness",
    "How many of these do you struggle with? #mentalhealthawareness #depressionandanxiety",
    "Mental health is an ongoing part of health, not something you 'fix' once. #mentalhealth #mentalwellness #comic",
    "CW: suicide. Crisis lines alone are not sufficient for suicide prevention but can be powerful resources in moments of mental health emergencies. #suicideprevention #mentalhealthawareness",
    "Boredom affects people with ADHD negatively to a larger extent. Encouraging movement and stimulation can help. #ADHD #mentalhealth #neurodiversity",
    "Drop three ♥♥♥ in the comments if you found this post helpful! Important things to know about passive suicidal ideation. #mentalhealthawareness #suicideprevention #mentalhealth"

]

# Sentiment analysis results
sentiment_results = {'Positive': 0, 'Neutral': 0, 'Negative': 0}

# Hashtag collection
hashtag_frequency = {}
hashtags_list = []  # List to collect hashtags for co-occurrence analysis

# 1. Sentiment and hashtag analysis
for post in posts:
    # VADER sentiment analysis
    sentiment_score = analyzer.polarity_scores(post)['compound']
    if sentiment_score >= 0.05:
        sentiment_type = 'Positive'
    elif sentiment_score <= -0.05:
        sentiment_type = 'Negative'
    else:
        sentiment_type = 'Neutral'
    sentiment_results[sentiment_type] += 1

    # Hashtag extraction
    hashtags = re.findall(r"#(\w+)", post)
    hashtags_list.append(hashtags)  # Collect hashtags for co-occurrence analysis
    for hashtag in hashtags:
        if hashtag in hashtag_frequency:
            hashtag_frequency[hashtag] += 1
        else:
            hashtag_frequency[hashtag] = 1

# 2. Graphical representation of sentiment results
def plot_sentiment_graph(sentiment_results):
    labels = list(sentiment_results.keys())
    sizes = list(sentiment_results.values())
    colors = ['#99ff99', '#66b3ff', '#ff9999']  # Green for positive, blue for neutral, red for negative
    explode = (0.1, 0, 0)  # Explode the 1st slice for visibility

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140)
    plt.axis('equal')
    plt.title('Sentiment analysis of Instagram posts', pad=25)
    plt.show()

# Plot the sentiment graph
plot_sentiment_graph(sentiment_results)

# 3. Hashtag analysis
def plot_hashtag_graph(hashtag_frequency):
    hashtags = list(hashtag_frequency.keys())
    counts = list(hashtag_frequency.values())

    # Sort by count in descending order
    sorted_hashtags = sorted(hashtag_frequency.items(), key=lambda x: x[1], reverse=True)
    hashtags, counts = zip(*sorted_hashtags)

    plt.figure(figsize=(10, 8))
    plt.barh(hashtags, counts, color='skyblue')
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Hashtags', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('Hashtag usage in Instagram posts', fontsize=14)
    plt.tight_layout()
    plt.show()

# Plot the hashtag frequency graph
plot_hashtag_graph(hashtag_frequency)

# 4. Hashtag co-occurrence analysis
# Get unique hashtags
unique_hashtags = list(set(itertools.chain(*hashtags_list)))

# Create an empty co-occurrence matrix
co_occurrence_matrix = pd.DataFrame(0, index=unique_hashtags, columns=unique_hashtags)

for hashtags in hashtags_list:
    for pair in itertools.combinations(hashtags, 2):
        co_occurrence_matrix.loc[pair[0], pair[1]] += 1
        co_occurrence_matrix.loc[pair[1], pair[0]] += 1

# Plot a heatmap of the co-occurrence matrix
plt.figure(figsize=(10, 8))
sns.heatmap(co_occurrence_matrix, annot=True, cmap="coolwarm", linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Hashtag co-occurrence heatmap', pad=20)
plt.tight_layout()
plt.show()

# Print summary
print(f"Sentiment analysis summary: {sentiment_results}")
print(f"Top hashtags: {sorted(hashtag_frequency.items(), key=lambda x: x[1], reverse=True)[:5]}")