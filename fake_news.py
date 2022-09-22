from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

## Reading data
FAKE = pd.read_csv("DataSet_Misinfo_FAKE.csv")
TRUE = pd.read_csv("DataSet_Misinfo_TRUE.csv")

FAKE.head()

FAKE.drop("Unnamed: 0", axis=1, inplace=True)

TRUE.head()

TRUE.drop("Unnamed: 0", axis=1, inplace=True)

## Creating labels
FAKE["IS_FAKE"] = 1

TRUE["IS_FAKE"] = 0

## Sampling from both because of memory issues
news = pd.concat([FAKE.sample(random_state=42, n=12000), TRUE.sample(random_state=42, n=12000)], ignore_index=True)

news.info()
news.head()

df = news.copy()

## Dropping nulls
df.dropna(how="any", inplace=True)

df.info()

df.columns = [col.upper() for col in df.columns]
df["TEXT"] = df["TEXT"].str.lower()

## Getting rid of punctuations and digits
df["TEXT"] = df["TEXT"].str.replace("[^\w\s]", "")
df["TEXT"] = df["TEXT"].str.replace("\d", "")

## Stop words
sw = stopwords.words('english')
df["TEXT"] = df["TEXT"].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

## Rare words
temp_df = pd.Series(' '.join(df["TEXT"]).split()).value_counts()
drops = temp_df[temp_df <= 10]
df["TEXT"] = df["TEXT"].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

## Lemmatization
df["TEXT"] = df["TEXT"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

tf = df["TEXT"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf.sort_values("tf", ascending=False)


# from sklearn.feature_extraction.text import CountVectorizer
#
# vectorizer = CountVectorizer()
# X_count = vectorizer.fit_transform(X)
# vectorizer.get_feature_names()


X = df["TEXT"]
y = df["IS_FAKE"]

from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_word_vectorizer = TfidfVectorizer()
X_tf_idf_word = tf_idf_word_vectorizer.fit_transform(X)

## SIA

sia = SentimentIntensityAnalyzer()

df["TEXT"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

df["POLARITY_SCORE"] = df["TEXT"].apply(lambda x: sia.polarity_scores(x)["compound"])

df.groupby("IS_FAKE").agg({"POLARITY_SCORE":"mean"})

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate, RandomizedSearchCV, validation_curve, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, plot_confusion_matrix

lr = LogisticRegression().fit(X_tf_idf_word, y)

cv_results = cross_validate(lr,
                            X_tf_idf_word, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall"])

cv_results['test_accuracy'].mean()
cv_results['test_precision'].mean()
cv_results['test_recall'].mean()

from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

rf_model = RandomForestClassifier(random_state=42).fit(X_tf_idf_word, y)

cv_results = cross_validate(rf_model,
                            X_tf_idf_word, y,
                            cv=5,
                            scoring=["f1", "accuracy", "precision", "recall", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_precision'].mean()
cv_results['test_recall'].mean()

rf_params = {"max_depth": [3, 5, 8, 12, None],
             "max_features": [3, 5, 7, 10, "sqrt"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [1200, 1500, 1700]}

# 1800, 2000, 2200, 2500

rf_best_grid = GridSearchCV(rf_model,
                              rf_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=1).fit(X_tf_idf_word, y)

rf_final = RandomForestClassifier(**rf_best_grid.best_params_, random_state=42).fit(X_tf_idf_word, y)

cv_results = cross_validate(rf_final,
                            X_tf_idf_word, y,
                            cv=5,
                            scoring=["precision", "recall", "accuracy"])

cv_results['test_accuracy'].mean()
cv_results['test_precision'].mean()
cv_results['test_recall'].mean()
cv_results["test_roc_auc"].mean()