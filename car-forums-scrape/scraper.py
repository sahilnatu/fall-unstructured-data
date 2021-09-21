# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 13:52:04 2021

@author: sahil

The following code looks to scrape the data on Entry Level Luxury Performance
Sedans from Edmunds forum, find similaritites between brands, and do further
analysis on the same
"""


from selenium import webdriver
import pandas as pd
from datetime import datetime
import nltk
import numpy as np
import matplotlib as plt
import math
from sklearn.manifold import MDS
#nltk.download('punkt')
#nltk.download('stopwords')

############# SCRAPER

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

wd = webdriver.Chrome('chromedriver',options=chrome_options)

# Open Link
link = "https://forums.edmunds.com/discussion/2864/general/x/entry-level-luxury-performance-sedans"
wd.get(link)

# Find Last Page
last_pg_anchortag_index = len(wd.find_elements_by_xpath('//*[@id="PagerBefore"]/a'))-1
last_pg = wd.find_elements_by_xpath('//*[@id="PagerBefore"]/a[{}]'.format(last_pg_anchortag_index))[0]
pg_no = last_pg.text

# Initialize an empty dataframe
comments = pd.DataFrame(columns = ['date','comments'])

# Scrape till we have enough comments
while len(comments)<5000:
    
    # Iterate through pages starting from latest comments (last page)
    link_updated = link+"/p"+pg_no
    wd.get(link_updated)
    
    # Remove block quotes in comments
    wd.execute_script('''
                      var element = document.getElementsByTagName("blockquote"), index;
                      for (index = element.length - 1; index >= 0; index--) {
                              element[index].parentNode.removeChild(element[index]);
                              }
                      ''')

    # Fetch list of all comments in a page
    elements = wd.find_elements_by_xpath('//ul[@class="MessageList DataList Comments pageBox"]/li')
    
    # Iterate through this list scraping only comment message and date, and storing it in the dataframe
    for i in range(len(elements)):
        date = datetime.fromisoformat(elements[0].find_elements_by_xpath('//div/div[2]/div[2]/span/a/time')[i].get_attribute('datetime')).date()
        comment = elements[0].find_elements_by_xpath('//div/div[3]/div/div[1]')[i].text
        comments = comments.append(pd.DataFrame([[date,comment]], columns=(['date','comments'])),ignore_index=True)
        
    # After scraping off the page, turnover to the previous page
    pg_no = str(int(pg_no)-1)
    
# Output the comments dataframe as a csv
comments.to_csv(r'edmunds.csv', index = False, header=True)

############## CONTINUE

comments = comments.fillna('')

# Extract each word in a comment
comments['words'] = np.nan
comments['car_mentions'] = np.nan
comments['non_car_mentions'] = np.nan
words_raw = []
words_clean = []
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stopwords = set(nltk.corpus.stopwords.words('english'))

for i in range(len(comments)):
    word_tokens = tokenizer.tokenize(comments['comments'].iloc[i].lower())
    word_tokens_clean = []
    for w in word_tokens:
        if w not in stopwords and w not in word_tokens_clean:
            word_tokens_clean.append(w)
    comments['words'].iloc[i] = word_tokens_clean
    words_clean = words_clean + word_tokens_clean
    words_raw = words_raw + nltk.tokenize.word_tokenize(comments['comments'].iloc[i].lower())
    
# Find the freq distribution of words and store 100 most common words in a dataframe
word_dist_raw = nltk.FreqDist(words_raw)
common_words_raw = pd.DataFrame(word_dist_raw.most_common(100),columns=(['word','freq']))

# Plot this frequency distribution for top 100 words on logarithmic x and y axes
log_x = []
for i in common_words_raw.index: log_x.append(math.log(i+1))
log_y = []
for i in common_words_raw['freq']: log_y.append(math.log(i))
plt.pyplot.figure(0)
plt.pyplot.plot(log_x, log_y, 'ro')
# Since the resulting graph has an almost straight line, we can say that the words in the scrapped data follow Zipf's Law

# Freq distribution of words without stopwords and punctuations    
word_dist_clean = nltk.FreqDist(words_clean)
common_words_clean = pd.DataFrame(word_dist_clean.most_common(),columns=(['word','freq']))

# Reading car brand-model master
car_master = pd.read_csv(r'car models and brands.csv')
car_master = car_master.drop_duplicates()

# Dropping models which have more than 1 brand against it
car_master_rem_dpl_models = car_master.groupby(['Model']).size().reset_index(name='count')
car_master_rem_dpl_models = car_master_rem_dpl_models[car_master_rem_dpl_models['count'] == 1]
car_master = car_master_rem_dpl_models.join(car_master.set_index('Model'), on='Model')[['Brand','Model']]

# Create list of brands and models in the car master data
brand_master_word_list = []
model_master_word_list = []
for i in range(len(car_master)):
    if car_master.iloc[i,0] not in brand_master_word_list:
        brand_master_word_list = brand_master_word_list + [car_master.iloc[i,0]]
    if car_master.iloc[i,1] not in model_master_word_list:
        model_master_word_list = model_master_word_list + [car_master.iloc[i,1]]
       
# Add mentions of brands and models (converted to brands) against each comment - do not count more than 1 mention of any brand in any given comment        
# Add all non car mentions against each comment
car_mentions_list = []
non_car_mentions_list = []
for i in range(len(comments)):
    car_words = []
    non_car_words = []
    for w in comments['words'].iloc[i]:
        if w in brand_master_word_list and w not in car_words:
            car_words = car_words + [w]
        elif w in model_master_word_list:
            brand = car_master[car_master['Model']==w]['Brand'].iloc[0]
            if brand not in car_words:
                car_words = car_words + [brand]
        elif w not in brand_master_word_list and w not in model_master_word_list and w not in non_car_words:
            non_car_words = non_car_words + [w]
    if len(car_words) > 0:
        comments['car_mentions'].iloc[i] = car_words
        car_mentions_list = car_mentions_list + car_words
    if len(non_car_words) > 0:
        comments['non_car_mentions'].iloc[i] = non_car_words
        non_car_mentions_list = non_car_mentions_list + non_car_words
  
# Freq distribution of car brands after cleaning
car_mentions_dist = nltk.FreqDist(car_mentions_list)
common_cars = pd.DataFrame(car_mentions_dist.most_common(),columns=(['brand','freq']))
common_cars_clean = common_cars[common_cars['brand'].apply(lambda brand: brand not in ['car','seat','sedan','problem'])]
common_cars_clean = common_cars_clean.iloc[:10]

# Freq distribution of non car words
non_car_mentions_dist = nltk.FreqDist(non_car_mentions_list)
common_features = pd.DataFrame(non_car_mentions_dist.most_common(),columns=(['feature','freq']))
common_features_clean = common_features[common_features['feature'].apply(lambda feature: feature in ['price','sport','miles','performance','engine','cost','luxury','awd','power','brand','premium','speed','tires','service','transmission','wheel','expensive','steering','manual','torque','suspension','leather','turbo','warranty','size','value','handling','wheels','rwd','v6','oil','tech','cheap','fast','fwd','mpg','maintenance','reliability','tire','gas','sports','prices','engines','quality','motor','mileage','light','door','reliable','fuel','mile','technology','automatic','brakes','pricing','family','diesel','hot','dsg','american','suv','weight','european','comfortable','design'])]

# Create Lift Table
comments['car_mentions'] = comments['car_mentions'].fillna('')
lift_table = pd.pivot_table(common_cars_clean, index='brand',columns='brand')
lift_table_index = lift_table.index.values.tolist()
for i in range(len(lift_table)):
    lift_table.iloc[i,i] = np.nan
    for j in range(len(lift_table)):
        if j>i:
            car_1 = lift_table_index[i]
            car_2 = lift_table_index[j]
            print(car_1," ",car_2)
            car_1_counter = 0
            car_2_counter = 0
            car_1_2_counter = 0
            for k in range(len(comments)):
                if car_1 in comments['car_mentions'].iloc[k] and car_2 in comments['car_mentions'].iloc[k]:
                    car_1_counter += 1
                    car_2_counter += 1
                    car_1_2_counter += 1
                elif car_1 in comments['car_mentions'].iloc[k]:
                    car_1_counter += 1
                elif car_2 in comments['car_mentions'].iloc[k]:
                    car_2_counter += 1
            lift_table.iloc[i,j] = (len(comments)*car_1_2_counter)/(car_1_counter*car_2_counter)

# Create Features Lift Table
comments['non_car_mentions'] = comments['non_car_mentions'].fillna('')
car_features = pd.DataFrame(columns = ['car','feature','lift'])
for i in range(len(common_cars_clean)):
    for j in range(len(common_features_clean)):
        car_counter = 0
        feature_counter = 0
        car_feature_counter = 0
        for k in range(len(comments)):
            if common_cars_clean['brand'].iloc[i] in comments['car_mentions'].iloc[k] and common_features_clean['feature'].iloc[j] in comments['non_car_mentions'].iloc[k]:
                car_counter += 1
                feature_counter += 1
                car_feature_counter += 1
            elif common_cars_clean['brand'].iloc[i] in comments['car_mentions'].iloc[k]:
                car_counter += 1
            elif common_features_clean['feature'].iloc[j] in comments['non_car_mentions'].iloc[k]:
                feature_counter += 1    
        lift_car_feature = (len(comments)*car_feature_counter)/(car_counter*feature_counter)
        car_features = car_features.append(pd.DataFrame([[common_cars_clean['brand'].iloc[i],common_features_clean['feature'].iloc[j],lift_car_feature]], columns=(['car','feature','lift'])),ignore_index=True)      
        
#car_features.to_csv(r'C:\Users\sahil\Desktop\carfeature.csv', index = False, header=True)
car_features = car_features.sort_values(['car','lift'],ascending=[True,False])
car_features['rank'] = car_features.groupby('car')['lift'].rank(method='max',ascending=False)
car_features = car_features[car_features['rank']<=5][['car','feature','lift']]
car_features.to_csv(r'carfeatureclean.csv', index = False, header=True)
        
# Convert Lift Table to Matrix
lift_matrix = lift_table.to_numpy()

# Plot MDS graph for lifts
mds = MDS(2,random_state=0)
for i in range(len(lift_matrix)):
    for j in range(len(lift_matrix)):
        if i==j:
            lift_matrix[i,j]=1
        elif i>j:
            lift_matrix[i,j] = 1/lift_matrix[j,i]
        elif i<j:
            lift_matrix[i,j] = 1/lift_matrix[i,j]

mds_lift_matrix = mds.fit_transform(lift_matrix)
x_mds = [row[0] for row in mds_lift_matrix]
y_mds = [row[1] for row in mds_lift_matrix]
plt.pyplot.figure(1)
plt.pyplot.scatter(x_mds,y_mds)
for i in range(len(lift_table_index)):
    plt.pyplot.annotate(lift_table_index[i], (x_mds[i],y_mds[i]))

