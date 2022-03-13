import pickle
from string import punctuation
from nltk.classify import NaiveBayesClassifier, accuracy
from nltk.corpus import movie_reviews, stopwords
from bs4 import BeautifulSoup
import requests
from nltk.tokenize import word_tokenize
from os.path import exists
import cv2
import numpy
import urllib.request
import matplotlib.pyplot as plt


def create_word_features(words):
    useful_words = [
        word for word in words if word not in stopwords.words("english") and word not in punctuation
    ]
    my_dict = dict([
        (word, True) for word in useful_words
    ])
    return my_dict


if not exists("model.pkl"):
    neg_reviews = []
    for fileid in movie_reviews.fileids('neg'):
        words = movie_reviews.words(fileid)
        neg_reviews.append((create_word_features(words), "negative"))

    pos_reviews = []
    for fileid in movie_reviews.fileids('pos'):
        words = movie_reviews.words(fileid)
        pos_reviews.append((create_word_features(words), "positive"))

    train_set = neg_reviews[:750] + pos_reviews[:750]
    test_set = neg_reviews[750:] + pos_reviews[750:]

    classifier = NaiveBayesClassifier.train(train_set)

    accuracy = accuracy(classifier, test_set)
    print(f'accuracy: {accuracy * 100}')

    with open('model.pkl', 'wb') as f:
        pickle.dump(classifier, f)


# kalo misal model.pkl dkd, jalanin si else:
else:
    with open('model.pkl', 'rb') as f:
        classifier = pickle.load(f)

        URL = "https://academicslc.github.io/E222-COMP6683-YT01-00/"
        r = requests.get(URL)

        soup = BeautifulSoup(r.content, 'html5lib')
        comments = soup.findAll('div', attrs={'class': 'user-post-content'})

        pos_or_neg = [
            classifier.classify(
                create_word_features(
                    word_tokenize(
                        comment.text.strip().lower()
                    )
                )
            )
            for comment in comments
        ]

        print(pos_or_neg)

        posts = soup.findAll('div', attrs={'class': 'user-post-container'})
        photos = [URL+post.findPreviousSibling('img')['src'] for post in posts]
        for photoUrl in photos:
            print(photoUrl)
            with urllib.request.urlopen(photoUrl) as url:
                arr = numpy.asarray(bytearray(url.read()), dtype=numpy.uint8)
                image = cv2.imdecode(arr, -1)

                R, G, B = cv2.split(image)

                output1_R = cv2.equalizeHist(R)
                output1_G = cv2.equalizeHist(G)
                output1_B = cv2.equalizeHist(B)

                equ = cv2.merge((output1_R, output1_G, output1_B))

                # stacking images side-by-side
                res = numpy.hstack((image, equ))

                f_name = photoUrl.replace(URL, '').replace(
                    'images/', '').replace('.jpg', '').replace('.jpeg', '')
                cv2.imshow(f_name, res)
                f_name = f'results/{f_name}.jpg'
                cv2.imwrite(f_name, res)

                # Display image depicting image intensity before and after histogram equalization
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                histogram_image = cv2.calcHist(
                    [gray_image], [0], None, [256], [0, 256]
                )
                gray_equ = cv2.cvtColor(equ, cv2.COLOR_BGR2GRAY)
                histogram_equ = cv2.calcHist(
                    [gray_equ], [0], None, [256], [0, 256]
                )
                plt.plot(histogram_image, color='k')
                plt.show()

                plt.plot(histogram_equ, color='k')
                plt.show()
