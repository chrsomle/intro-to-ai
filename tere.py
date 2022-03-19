import pickle  # krn hrus save jdi model pickle
from string import punctuation  # u/ hapus tanda baca
# u/ klasifikasi model naive dan test akurasi
from nltk.classify import NaiveBayesClassifier, accuracy
# u/ membuat create word features biar identifikasi movie review pos/nef dan kata yg ga guna
from nltk.corpus import movie_reviews, stopwords
from bs4 import BeautifulSoup  # u/ scarping dr web, krn pke web mkannya jd html
from requests import get  # untuk load content web
from nltk.tokenize import word_tokenize  # u/ memilah word yang diperlukan
from os.path import exists  # u/ lihat model kita ada atau ga di file saya
import cv2  # u/ untuk manipulasi gambar
import numpy  # u/ decode dan encode foto krn laptop ga bs baca gambar, tp angka
import urllib.request  # u/ load image, pembukaan sebelum encode dan decode
# u/ rumus menunjukkan foto menggunakan warna hitam (k)
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
    neg_reviews = [
        (create_word_features(movie_reviews.words(fileid)), "negative")
        for fileid in movie_reviews.fileids('neg')
    ]
    print(len(neg_reviews))  # 1000

    pos_reviews = [
        (create_word_features(movie_reviews.words(fileid)), "positive")
        for fileid in movie_reviews.fileids('pos')
    ]
    print(len(pos_reviews))  # 1000 -> 75%
    # masing" neg_reviews dan pos_reviews berisi 1000 review data
    # ambil 750 data pertama dari masing" neg_reviews dan pos_reviews
    train_set = neg_reviews[:750] + pos_reviews[:750]
    # ambil data dimulai dari urutan ke 750 sampai terakhir.
    test_set = neg_reviews[750:] + pos_reviews[750:]

    classifier = NaiveBayesClassifier.train(train_set)

    accuracy = accuracy(classifier, test_set)
    print(f'accuracy: { accuracy * 100 }')

    with open('model.pkl', 'wb') as model:
        pickle.dump(classifier, model)


# kalo misal model.pkl dkd, jalanin si else:
else:
    with open('model.pkl', 'rb') as model:
        classifier = pickle.load(model)

        URL = "https://academicslc.github.io/E222-COMP6683-YT01-00/"
        response = get(URL)

        scrape = BeautifulSoup(response.content, 'html5lib')
        comments = scrape.findAll('div', attrs={'class': 'user-post-content'})

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

        posts = scrape.findAll('div', attrs={'class': 'user-post-container'})
        print(posts)
        photos = [
            # Example: https://academicslc.github.io/E222-COMP6683-YT01-00/ + images/dwayne_johnson.jpeg
            (URL)+(post.findPreviousSibling('img')['src'])
            for post in posts
        ]
        print(photos)

        for photoUrl in photos:
            print(photoUrl)
            with urllib.request.urlopen(photoUrl) as url:
                # mengubah gambar jadi angka / nge-decode image menjadi data type integer (angka)
                arr = numpy.asarray(bytearray(url.read()),
                                    dtype=numpy.uint8)  # encode
                image = cv2.imdecode(arr, -1)  # decode

                R, G, B = cv2.split(image)  # exract komponen gambar

                output1_R = cv2.equalizeHist(R)
                output1_G = cv2.equalizeHist(G)
                output1_B = cv2.equalizeHist(B)

                equ = cv2.merge((output1_R, output1_G, output1_B))

                # stacking images side-by-side
                result = numpy.hstack((image, equ))  # horizontal stack

                f_name = photoUrl.replace(URL, '').replace(
                    'images/', '').replace('.jpg', '').replace('.jpeg', '')  # emg bisa di simplify pake RegEx tp mager

                cv2.imshow(f_name, result)

                # masukin ke folder/directory namanya 'results'
                f_name = f'results/{f_name}.jpg'

                cv2.imwrite(f_name, result)

                # Display image depicting image intensity before and after histogram equalization
                # convert color ke gray
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                histogram_image = cv2.calcHist(
                    [gray_image], [0], None, [256], [0, 256]
                )
                gray_equ = cv2.cvtColor(equ, cv2.COLOR_BGR2GRAY)
                histogram_equ = cv2.calcHist(
                    [gray_equ], [0], None, [256], [0, 256]
                )
                # k itu black, makanya diagramnya wrna hitam
                plt.plot(histogram_image, color='k')
                plt.show()

                plt.plot(histogram_equ, color='k')
                plt.show()
