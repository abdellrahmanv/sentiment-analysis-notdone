{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# IMDB Movie Reviews Sentiment Analysis - Exploratory Data Analysis\n",
    "\n",
    "This notebook explores the IMDB dataset of 50k movie reviews for sentiment analysis."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from wordcloud import WordCloud\n",
    "import re\n",
    "from collections import Counter\n",
    "import nltk\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.tokenize import word_tokenize\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Download NLTK data\n",
    "nltk.download('punkt')\n",
    "nltk.download('stopwords')\n",
    "\n",
    "plt.style.use('ggplot')\n",
    "sns.set_palette(\"husl\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Data Loading and Basic Information"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the dataset\n",
    "df = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')\n",
    "\n",
    "print(f\"Dataset shape: {df.shape}\")\n",
    "print(f\"\\nColumn names: {list(df.columns)}\")\n",
    "print(f\"\\nData types:\")\n",
    "print(df.dtypes)\n",
    "print(f\"\\nFirst few rows:\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check for missing values\n",
    "print(\"Missing values:\")\n",
    "print(df.isnull().sum())\n",
    "\n",
    "# Check data info\n",
    "print(\"\\nDataset Info:\")\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Sentiment Distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Sentiment distribution\n",
    "sentiment_counts = df['sentiment'].value_counts()\n",
    "print(\"Sentiment distribution:\")\n",
    "print(sentiment_counts)\n",
    "print(f\"\\nPercentage distribution:\")\n",
    "print(df['sentiment'].value_counts(normalize=True) * 100)\n",
    "\n",
    "# Visualization\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))\n",
    "\n",
    "# Bar plot\n",
    "sentiment_counts.plot(kind='bar', ax=ax1, color=['skyblue', 'lightcoral'])\n",
    "ax1.set_title('Distribution of Sentiments')\n",
    "ax1.set_xlabel('Sentiment')\n",
    "ax1.set_ylabel('Count')\n",
    "ax1.tick_params(axis='x', rotation=0)\n",
    "\n",
    "# Pie chart\n",
    "ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', \n",
    "        colors=['skyblue', 'lightcoral'])\n",
    "ax2.set_title('Sentiment Distribution')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Text Length Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate text lengths\n",
    "df['review_length'] = df['review'].str.len()\n",
    "df['word_count'] = df['review'].str.split().str.len()\n",
    "\n",
    "# Statistics by sentiment\n",
    "print(\"Text length statistics by sentiment:\")\n",
    "print(df.groupby('sentiment')[['review_length', 'word_count']].describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize text length distributions\n",
    "fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n",
    "\n",
    "# Character length distribution\n",
    "df[df['sentiment'] == 'positive']['review_length'].hist(bins=50, alpha=0.7, \n",
    "                                                        label='Positive', ax=axes[0,0])\n",
    "df[df['sentiment'] == 'negative']['review_length'].hist(bins=50, alpha=0.7, \n",
    "                                                        label='Negative', ax=axes[0,0])\n",
    "axes[0,0].set_title('Distribution of Review Length (Characters)')\n",
    "axes[0,0].set_xlabel('Number of Characters')\n",
    "axes[0,0].set_ylabel('Frequency')\n",
    "axes[0,0].legend()\n",
    "\n",
    "# Word count distribution\n",
    "df[df['sentiment'] == 'positive']['word_count'].hist(bins=50, alpha=0.7, \n",
    "                                                     label='Positive', ax=axes[0,1])\n",
    "df[df['sentiment'] == 'negative']['word_count'].hist(bins=50, alpha=0.7, \n",
    "                                                     label='Negative', ax=axes[0,1])\n",
    "axes[0,1].set_title('Distribution of Word Count')\n",
    "axes[0,1].set_xlabel('Number of Words')\n",
    "axes[0,1].set_ylabel('Frequency')\n",
    "axes[0,1].legend()\n",
    "\n",
    "# Box plots\n",
    "df.boxplot(column='review_length', by='sentiment', ax=axes[1,0])\n",
    "axes[1,0].set_title('Review Length by Sentiment')\n",
    "axes[1,0].set_xlabel('Sentiment')\n",
    "axes[1,0].set_ylabel('Review Length')\n",
    "\n",
    "df.boxplot(column='word_count', by='sentiment', ax=axes[1,1])\n",
    "axes[1,1].set_title('Word Count by Sentiment')\n",
    "axes[1,1].set_xlabel('Sentiment')\n",
    "axes[1,1].set_ylabel('Word Count')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Word Cloud Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to clean text for word cloud\n",
    "def clean_text(text):\n",
    "    # Remove HTML tags\n",
    "    text = re.sub(r'<.*?>', '', text)\n",
    "    # Remove special characters and digits\n",
    "    text = re.sub(r'[^a-zA-Z\\s]', '', text)\n",
    "    # Convert to lowercase\n",
    "    text = text.lower()\n",
    "    return text\n",
    "\n",
    "# Clean reviews\n",
    "df['cleaned_review'] = df['review'].apply(clean_text)\n",
    "\n",
    "# Separate positive and negative reviews\n",
    "positive_reviews = ' '.join(df[df['sentiment'] == 'positive']['cleaned_review'])\n",
    "negative_reviews = ' '.join(df[df['sentiment'] == 'negative']['cleaned_review'])\n",
    "\n",
    "# Create word clouds\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))\n",
    "\n",
    "# Positive word cloud\n",
    "wordcloud_pos = WordCloud(width=800, height=400, background_color='white',\n",
    "                          max_words=100, colormap='Blues').generate(positive_reviews)\n",
    "ax1.imshow(wordcloud_pos, interpolation='bilinear')\n",
    "ax1.set_title('Most Common Words in Positive Reviews', fontsize=16)\n",
    "ax1.axis('off')\n",
    "\n",
    "# Negative word cloud\n",
    "wordcloud_neg = WordCloud(width=800, height=400, background_color='white',\n",
    "                          max_words=100, colormap='Reds').generate(negative_reviews)\n",
    "ax2.imshow(wordcloud_neg, interpolation='bilinear')\n",
    "ax2.set_title('Most Common Words in Negative Reviews', fontsize=16)\n",
    "ax2.axis('off')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Most Common Words Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get stopwords\n",
    "stop_words = set(stopwords.words('english'))\n",
    "\n",
    "def get_top_words(text, n=20):\n",
    "    words = word_tokenize(text.lower())\n",
    "    words = [word for word in words if word.isalpha() and word not in stop_words]\n",
    "    return Counter(words).most_common(n)\n",
    "\n",
    "# Get top words for each sentiment\n",
    "top_positive = get_top_words(positive_reviews)\n",
    "top_negative = get_top_words(negative_reviews)\n",
    "\n",
    "print(\"Top 20 words in positive reviews:\")\n",
    "for word, count in top_positive:\n",
    "    print(f\"{word}: {count}\")\n",
    "\n",
    "print(\"\\nTop 20 words in negative reviews:\")\n",
    "for word, count in top_negative:\n",
    "    print(f\"{word}: {count}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize top words\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))\n",
    "\n",
    "# Positive words\n",
    "pos_words, pos_counts = zip(*top_positive)\n",
    "ax1.barh(range(len(pos_words)), pos_counts, color='skyblue')\n",
    "ax1.set_yticks(range(len(pos_words)))\n",
    "ax1.set_yticklabels(pos_words)\n",
    "ax1.set_title('Top 20 Words in Positive Reviews')\n",
    "ax1.set_xlabel('Frequency')\n",
    "ax1.invert_yaxis()\n",
    "\n",
    "# Negative words\n",
    "neg_words, neg_counts = zip(*top_negative)\n",
    "ax2.barh(range(len(neg_words)), neg_counts, color='lightcoral')\n",
    "ax2.set_yticks(range(len(neg_words)))\n",
    "ax2.set_yticklabels(neg_words)\n",
    "ax2.set_title('Top 20 Words in Negative Reviews')\n",
    "ax2.set_xlabel('Frequency')\n",
    "ax2.invert_yaxis()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Sample Reviews Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Display sample reviews\n",
    "print(\"Sample Positive Reviews:\")\n",
    "print(\"=\" * 50)\n",
    "for i, review in enumerate(df[df['sentiment'] == 'positive']['review'].head(3)):\n",
    "    print(f\"Review {i+1}:\")\n",
    "    print(review[:500] + \"...\" if len(review) > 500 else review)\n",
    "    print(\"-\" * 30)\n",
    "\n",
    "print(\"\\nSample Negative Reviews:\")\n",
    "print(\"=\" * 50)\n",
    "for i, review in enumerate(df[df['sentiment'] == 'negative']['review'].head(3)):\n",
    "    print(f\"Review {i+1}:\")\n",
    "    print(review[:500] + \"...\" if len(review) > 500 else review)\n",
    "    print(\"-\" * 30)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Data Quality Insights"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check for HTML tags in reviews\n",
    "html_count = df['review'].str.contains('<.*?>', regex=True).sum()\n",
    "print(f\"Number of reviews containing HTML tags: {html_count}\")\n",
    "\n",
    "# Check for very short/long reviews\n",
    "very_short = (df['word_count'] < 10).sum()\n",
    "very_long = (df['word_count'] > 1000).sum()\n",
    "print(f\"Reviews with less than 10 words: {very_short}\")\n",
    "print(f\"Reviews with more than 1000 words: {very_long}\")\n",
    "\n",
    "# Check for duplicate reviews\n",
    "duplicates = df['review'].duplicated().sum()\n",
    "print(f\"Number of duplicate reviews: {duplicates}\")\n",
    "\n",
    "# Summary statistics\n",
    "print(f\"\\nSummary Statistics:\")\n",
    "print(f\"Total reviews: {len(df)}\")\n",
    "print(f\"Average review length: {df['review_length'].mean():.2f} characters\")\n",
    "print(f\"Average word count: {df['word_count'].mean():.2f} words\")\n",
    "print(f\"Median review length: {df['review_length'].median():.2f} characters\")\n",
    "print(f\"Median word count: {df['word_count'].median():.2f} words\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}