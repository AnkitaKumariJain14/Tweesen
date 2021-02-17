# Tweesen
Team: Trojan_Horse

The Project processes and detects tweets collected from twitter and divides them into 6 classes - Positive, Negative, Toxic, Obscene, Threat and Insult.
Extraction of tweets from twitter is done by the use of Selenium driver and modules to access Twitter and search for verified profiles of subjects to scrape the top tweets by the subject. 
This data is written into a .CSV file. 
All these tweets are given as input to 3 models – Twitter Sentiment analysis model, Toxic classification and the Big5 model after preprocessing by removal of stop words and unwanted characters followed by Tokenization. 
The output obtained is written into two files – Information.txt and Prediction.txt

## Dataset: 
1. Jigsaw Toxic Comments - https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
2. Big 5 Personality Detection - https://github.com/SenticNet/personality-detection/tree/67c97e7d9353d036304305709cdca8eef8dcc6ad
3. Sentiment140 dataset with 1.6 million tweets - https://www.kaggle.com/kazanova/sentiment140

## How to run the code:
1. Install all the modules mentioned here: 

          - Keras==2.4.3
	  - Keras-Preprocessing==1.1.2
	  - msedge-selenium-tools==3.141.3
	  - nltk ==3.5
	  - numpy==1.18.5
	  - pandas==1.2.1
         - pickleshare==0.7.5
	  - pickle5==0.0.11
	  - selenium==3.141.0
	  - snowballstemmer==2.0.0
	  - tensorflow==2.4.1
	  - wikipedia==1.4.0
2. According to the given input format, pass the input.txt file that contains name of the subject, profession, country etc., along with the number of test cases as "py final.py --testcase *no of testcases* --inputfile "input.txt"" 
3. After the code execution, the following files will be saved in the directory:
	Information.txt: 
	Information.txt contains the name followed by the query and a wikipedia summary of the subject. It also contains all the tweets that were taken into account for processing by various models.
	Prediction.txt: 
	This file contains the query followed by number of positive, negative, Toxic, Obscene, Threat and Insult tweets. It is followed by the Big 5 classification of the subject. 

## Further Development: 
1. Incorporation of Tweets in various languages to increase the scope of detection.
2. Including other social media platforms and blogs. 
3. Using other factors derived from the tweets like likes, comments, the reach obtained etc to have better classfication of data. 
4. Including non-verified accounts for prediction. 

*Disclaimer: Smooth functioning of the code requires good internet connectivity. Re-run the code in case of error.*
