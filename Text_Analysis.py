
# coding: utf-8

# In[13]:


from selenium import webdriver
from bs4 import BeautifulSoup
import os
import time
import nltk


# In[14]:


#url to get reviews
url = "https://www.yelp.com/biz/pequods-pizzeria-chicago?osq=Restaurants"

css_last_page_number_link_selector = ".review-pager .pagination-block .page-of-pages.arrange_unit.arrange_unit--fill"
css_review_selector = ".review-wrapper .review-content > p"
css_rating_selector = ".review-wrapper .review-content .biz-rating.biz-rating-large.clearfix > div > div"
css_next_page_button_selector = ".review-pager .pagination-block .pagination-links.arrange_unit > div > div > a.next"


# In[15]:


#create a new Chrome session
chromedriver = "C:/Users/adity/Downloads/chromedriver.exe"
os.environ["webdriver.chrome.driver"] = chromedriver
driver = webdriver.Chrome(chromedriver)
driver.get(url)


# In[16]:


def get_page_numbers():
    time.sleep(3)
    page = driver.page_source
    soup = BeautifulSoup(page)
    page_of_pages = soup.select(css_last_page_number_link_selector)[-1].get_text()
    last_page = page_of_pages.strip().split(' ')[-1]
    return last_page


# In[17]:


# Save extracted info
total_pages = int(get_page_numbers())
scraped_data = []
scraped_ratings = []


# In[18]:


# Get the reviews from website
def extract_reviews():
    time.sleep(2)
    page = driver.page_source
    soup = BeautifulSoup(page)
    parent_div = soup.select(css_review_selector)
    for child_element in parent_div:
        scraped_data.append(child_element.get_text())


# In[19]:


# Get ratings for review
def extract_ratings():
    page = driver.page_source
    soup = BeautifulSoup(page)
    rating_div = soup.select(css_rating_selector)
    for rating in rating_div:
        rating_in_title = rating["title"]
        rating_str = rating_in_title.strip().split(' ')[0]
        scraped_ratings.append(float(rating_str))


# In[20]:


# Go to next page
def goToNextPage():
    driver.find_element_by_css_selector(css_next_page_button_selector).click()


# In[21]:


def get_all_reviews():
    i = 0
    while i < total_pages:
        extract_reviews()
        extract_ratings()
        time.sleep(3)
        i = i + 1
        goToNextPage()

get_all_reviews()


# In[22]:


# Lemmatizating reviews
from nltk.stem import WordNetLemmatizer

lemmatized_reviews = []
lemmatizer = WordNetLemmatizer()

def lemmatize_review(paragraph):
    sentences = nltk.sent_tokenize(paragraph)
    for i in range(len(sentences)):
        words = nltk.word_tokenize(sentences[i])
        newwords = []
        for word in words:
            newwords.append(lemmatizer.lemmatize(word))
        sentences[i] = ' '.join(newwords)
    paragraph = '.'.join(sentences)
    lemmatized_reviews.append(paragraph)


# In[23]:


def process_reviews():
    for data in scraped_data:
        lemmatize_review(data)

process_reviews()


# In[24]:


# Importing negative and positive words
def isNotNull(value):
    return value is not None and len(value)>0

dict_pos = []
dict_neg = []

f = open('negative-words.txt','r')

for line in f:
    t= line.strip().lower();
    if (isNotNull(t)):
        dict_neg.append(t)
f.close()

f = open('positive-words.txt','r')
for line in f:
    t = line.strip().lower();
    if (isNotNull(t)):
        dict_pos.append(t)
f.close()


# In[25]:


# Calculate positive and negative review based on ratings
pos_content_rating = []
neg_content_rating = []

def calculate_review_rating(sentence, rating):
    if rating > 3.0 and rating <= 5.0:
        pos_content_rating.append(sentence)
    elif rating < 3.0:
        neg_content_rating.append(sentence)

for i in range(len(scraped_ratings)):
    calculate_review_rating(scraped_data[i], scraped_ratings[i])


# In[26]:


# Calculate positive and negative review based on sentiments
pos_content_sent_score = []
neg_content_sent_score = []
    
def calculate_sentiment(sentence):
    pos_cnt = 0
    neg_cnt = 0
    words = nltk.word_tokenize(sentence)
    for word in words:
        if word in dict_pos:
            pos_cnt = pos_cnt + 1
        if word in dict_neg:
            neg_cnt = neg_cnt + 1
    neg_content_sent_score.append(neg_cnt)
    pos_content_sent_score.append(pos_cnt)
    
for data in scraped_data:
    calculate_sentiment(data)


# In[27]:


# Saving the separated reviews
import pandas as pd
import csv


# In[28]:


# Create a directory to store files
cwd = os.getcwd()
directory = cwd + '/dataset'
if not os.path.exists(directory):
    os.makedirs(directory)

write_or_create_positive_file = 'w'
if not os.path.exists(directory + '/positive_ratings.csv'):
    write_or_create_positive_file = 'x'

with open(directory + '/positive_ratings.csv', write_or_create_positive_file, newline='') as myfile:
    wr = csv.writer(myfile, dialect = 'excel', quoting=csv.QUOTE_ALL)
    for content in pos_content_rating:
        wr.writerow([content])


write_or_create_negative_file = 'w'
if not os.path.exists(directory + '/negative_ratings.csv'):
    write_or_create_negative_file = 'x'

with open(directory + '/negative_ratings.csv', write_or_create_positive_file, newline='') as myfile:
    wr = csv.writer(myfile,  dialect = 'excel', quoting=csv.QUOTE_ALL)
    for neg_sent in neg_content_rating:
        wr.writerow([neg_sent])


# In[29]:


# Creating dataset for algorithm
df = pd.DataFrame()
y_text = []
y = []
for data in scraped_data:
    if data in pos_content_rating:
        y.append(1)
        y_text.append('positive')
    else:
        y.append(0)
        y_text.append('negative')
    
df['positive_score'] = pos_content_sent_score
df['negative_score'] = neg_content_sent_score


# In[30]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[31]:


model = MultinomialNB()
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

model.fit(X_train, y_train)

predicted= model.predict(X_test)

print(confusion_matrix(y_test, predicted))

print(classification_report(y_test, predicted))


# In[32]:


dataset = pd.DataFrame()
dataset['reviews'] = scraped_data
dataset['positive_score'] = pos_content_sent_score
dataset['negative_score'] = neg_content_sent_score
dataset['result'] = y_text


# In[33]:


dataset.to_csv('./dataset/text_analytics_dataset.csv')


# In[34]:


df_pred = pd.DataFrame()
df_pred = X_test
df_pred['result'] = y_test
df_pred['predicted'] = predicted
df_pred.to_csv('./dataset/predicted_output_with_actual_result.csv')


# In[ ]:




