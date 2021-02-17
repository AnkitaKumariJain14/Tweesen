### Importing the libraries ###
import sys
import re
import csv 
import pandas as pd
import pickle
import wikipedia
from getpass import getpass
from time import sleep
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from msedge.selenium_tools import Edge, EdgeOptions
from selenium.webdriver.common.action_chains import ActionChains
import tensorflow as tf
import numpy as np
import re
from tensorflow import keras
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from keras.preprocessing.sequence import pad_sequences

stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')
text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

### Helper Functions ###
def main():
    args=sys.argv
    f = open(args[4], "r")
    Lines = f.readlines()
    names,profession,nationality,job =[],[],[],[]
    for line in Lines:
        array = line.split(",")
        names.append(array[0])
        profession.append(array[1])
        nationality.append(array[2])
        job.append(array[3].replace("\n",""))
    for name in names: 
        print("Query:", name, ".\nProcessing...")
        user = 'SussanMicheal'
        search_term = f'{name} filter:verified'
        options = EdgeOptions()
        options.use_chromium = True
        driver = Edge(options=options)
        driver.get('https://www.twitter.com/login')
        driver.maximize_window()
        sleep(2)
        username = driver.find_element_by_xpath('//input[@name="session[username_or_email]"]')
        username.send_keys(user)
        password = driver.find_element_by_xpath('//input[@name="session[password]"]')
        password.send_keys('donkey123')
        password.send_keys(Keys.RETURN)
        sleep(1)
        search_input = driver.find_element_by_xpath('//input[@aria-label="Search query"]')
        search_input.send_keys(search_term)
        search_input.send_keys(Keys.RETURN)
        sleep(1)
        driver.find_element_by_link_text('People').click()
        sleep(3)
        driver.find_element_by_xpath('//div[@class="css-1dbjc4n r-j7yic r-qklmqi r-1adg3ll r-1ny4l3l"]').click()
        sleep(3)
        data = []
        tweet_data =[]
        start = 0
        end =500
        for i in range(0,5):
            sleep(1)
            cards = driver.find_elements_by_xpath('//div[@data-testid="tweet"]')
            card = cards[i]
            tweet = get_tweet_data(card)
            for card in cards:
                data = get_tweet_data(card)
                if data:
                    tweet_data.append(data) 
            driver.execute_script(f'window.scrollTo({start},{end});')
            start += 500
            end += 500
        driver.close()
        tweets=set(tweet_data)
        write_to_csv(name,tweets)
        df = pd.read_csv(f'{name}.csv')
        Twitter_sentiment=Twitter_sentiment_model(df)
        Twitter_toxic=Twitter_toxic_model(df)
        Big5 =Big5_model(df)

        create_report(name, tweets, Twitter_sentiment, Twitter_toxic, Big5)

def get_tweet_data(card):
   
    username = card.find_element_by_xpath('.//span').text
    try:
        handle = card.find_element_by_xpath('.//span[contains(text(), "@")]').text
    except NoSuchElementException:
        return
    
    try:
        postdate = card.find_element_by_xpath('.//time').get_attribute('datetime')
    except NoSuchElementException:
        return
    
    comment = card.find_element_by_xpath('.//div[2]/div[2]/div[1]').text
    responding = card.find_element_by_xpath('.//div[2]/div[2]/div[2]').text
    text = comment + responding
    reply_cnt = card.find_element_by_xpath('.//div[@data-testid="reply"]').text
    retweet_cnt = card.find_element_by_xpath('.//div[@data-testid="retweet"]').text
    like_cnt = card.find_element_by_xpath('.//div[@data-testid="like"]').text
    
    tweets = (username, handle, postdate, text,reply_cnt, retweet_cnt, like_cnt)
    return tweets

def write_to_csv(name,tweets):
    with open(f'{name}.csv', 'w', newline='', encoding='utf-8') as f:
        header = ['UserName', 'Handle', 'Timestamp', 'Text', 'Comments', 'Likes', 'Retweets']
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(tweets)
def create_report(name, tweets, Twitter_sentiment, Twitter_toxic, Big5):
    print("Writing report...")

    with open('information.txt', 'a',encoding='utf8', errors='ignore') as f:
        f.writelines(["QUERY : ", name, "\n\n"])
        f.writelines([wikipedia.summary(f"{name}"),"\n\n"])
        with open(f'{name}.csv', "r" ,encoding='utf8', errors='ignore') as df:
            dataframe=pd.read_csv(df)
            user_name=dataframe.Handle[0]
            f.writelines(f"Twitter Handler: www.twitter.com/{user_name.replace('@','')}\n\n")
            f.writelines('Recent Tweets: '+'\n\n' )
            for row in dataframe.Text:
                f.write(str(row) +'\n')
    f.close()  

    with open('prediction.txt', 'a', encoding='utf8', errors='ignore') as f:
        f.writelines(["QUERY : ", name, "\n\n"])
        f.writelines(["Personality analysis: \n\n", Big5, "\n\n"])
        f.writelines(["Sentiment analysis of  ", str(len(tweets)), " tweets: \n", str(Twitter_sentiment.count('Positive')), " are positive and ", str(Twitter_sentiment.count('Negative')), " are negative.\n\n"])
        f.writelines(["Further analysis of  ", str(len(tweets))," tweets: \n"])
        f.writelines([str(np.count_nonzero(Twitter_toxic[0])), " are toxic, \n"])
        f.writelines([str(np.count_nonzero(Twitter_toxic[1])), " contain obscene text, \n"])
        f.writelines([str(np.count_nonzero(Twitter_toxic[2])), " are threats, \n"])
        f.writelines([str(np.count_nonzero(Twitter_toxic[3])), " are insults. \n\n\n"])
    f.close()
    print("Data and report generated.")
    

def preprocess(text, stem=False):
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

def Twitter_sentiment_model (df):
    sentiment_twitter_model = keras.models.load_model('Twitter_model.h5')
    df.Text = df.Text.apply(lambda x: preprocess(x))

    with open('sajid_tokenizer.pickle', 'rb') as f:
        sentiment_model_tokenizer = pickle.load(f)
    x_train1 = pad_sequences(sentiment_model_tokenizer.texts_to_sequences(df.Text),
                        maxlen = 30)

    pred1= []
    for i in range(len(df)):
        pred1.append(sentiment_twitter_model.predict(x_train1)[i])

    def decode_sentiment(score):
        return "Positive" if score>0.3 else "Negative"
    y_pred= [decode_sentiment(pred) for pred in pred1]
    return y_pred

def Twitter_toxic_model (df):
    toxic_model = keras.models.load_model('Ankimodel.h5')
    df.Text = df.Text.apply(lambda x: preprocess(x))

    with open('Anki_tokenizer.pickle', 'rb') as f:
        toxic_model_tokenizer = pickle.load(f)

    x_train2 = pad_sequences(toxic_model_tokenizer.texts_to_sequences(df.Text),
                        maxlen = 30)

    def decode_sentiment1(score):
                    result= []
                    if score[0]>0.5:
                        result.append("Toxic")
                    else:
                        result.append("-")
                    if score[1]>0.5:
                        result.append("Obscene")
                    else:
                        result.append("-")
                    if score[2]>0.5:
                        result.append("Threat")
                    else:
                        result.append("-")
                    if score[3]>0.5:
                        result.append("Insult")
                    else:
                        result.append("-")
                    return result

    scores = toxic_model.predict(x_train2, verbose=1, batch_size=10000)
    y_prediction = (scores>0.5).astype('int64').T

    return y_prediction

def Big5_model (df):
    big5_model = keras.models.load_model('big5g.h5')
    df.Text = df.Text.apply(lambda x: preprocess(x))

    with open('Ayisha_tokenizer.pickle', 'rb') as f:
        big5_model_tokenizer = pickle.load(f)
    x_train3 = pad_sequences(big5_model_tokenizer.texts_to_sequences(["".join([i for i in df.Text])]),
                            maxlen = 100)
        
    scores = big5_model.predict(x_train3, verbose=1, batch_size=10000)
    
    scores = scores[0]

    big5_targets = [['outgoing/energetic', 'solitary/reserved', 'Extraversion'],
                ['sensitive/nervous', 'resilient/confident', 'Neuroticism'],
                ['friendly/compassionate', 'challenging/callous', 'Agreeableness'],
                ['efficient/organized', 'extravagant/careless', 'Conscientiousness'],
                ['inventive/curious', 'consistent/cautious', 'Openness to experience']]

    big5_outputs = ""

    for i in range(len(scores)):
        big5_outputs = big5_outputs + big5_targets[i][2] + ' : ' + big5_targets[i][int(scores[i] > 0.5)] + "\n"

    return big5_outputs

if __name__ == '__main__':
    main()


