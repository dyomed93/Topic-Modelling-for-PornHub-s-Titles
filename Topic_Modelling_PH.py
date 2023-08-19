import re
from urllib.request import urlopen
from tqdm import tqdm
from bs4 import BeautifulSoup
import pandas as pd
from lingua import Language, LanguageDetectorBuilder
import pyLDAvis.sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

def clean_text(content):
    '''
    Regular expression that removes links and special characters from tweet.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(https?\S+)", " ", content).split())

def get_title(soup, lista_rip):
    '''

    Obtain Video Titles
    '''
    title = soup.find_all("span", {
        "class": "title"})
    for t in tqdm(title):
        t = str(t)
        l = detector.detect_language_of(t)
        try:
            l = str(l).split(".")[1][0:2].lower()
        except:
            pass
        t = re.sub(pattern_order, '', t)
        if l != "it":
            try:
                t = clean_text(t.split('title="')[1].split('">')[0]).lower()
                lista_rip.append(t)
            except:
                pass

def get_rating(soup, lista_rip):
    '''
    Obtain Video Ratings
    '''
    rat = soup.find_all("div", {
        "class": "value"})
    for t in tqdm(rat):
        t = str(t)
        try:
            t = clean_text(t.split('value"')[1].split('div')[0]).lower()
            lista_rip.append(t)
        except:
            pass

def get_views(soup, lista_rip):
    '''
    Obtain Video Views
    '''
    views = soup.find_all("span", {
        "class": "views"})
    for view in tqdm(views):
        view = str(view)
        try:
            view = view.split('var>')[1].split('</var>')[0].split('</')[0]
            lista_rip.append(view)
        except:
            pass

def get_time(soup, lista_rip):
    '''
    Obtain Video Length
    '''
    times = soup.find_all("var", {
        "class": "added"})
    for t in tqdm(times):
        t = str(t)
        try:
            t = t.split('"added">')[1].split('</var>')[0]
            lista_rip.append(t)
        except:
            pass

def display_topics(model, feature_names, no_top_words):
    """
        Displays found topics
    """
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
            for i in topic.argsort()[:-no_top_words - 1:-1]]))

#Lists and Variables Definition for Scraping
headers = {"User-Agent" : "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, Like Gecko) "
                          "Chrome/47.0.2526.106 Safari/537.36 "}
url = "https://www.pornhub.com/video"
url_female = "https://www.pornhub.com/popularwithwomen"
link_next_page = "?page=" #aggiungi il numero progressivamente
titrip = []
rating = []
views = []
time = []
titrip_fem = []
rating_fem = []
views_fem = []
time_fem = []

df_title= pd.DataFrame
i = 1
detector = LanguageDetectorBuilder.from_languages(Language.ITALIAN, Language.ENGLISH).with_minimum_relative_distance(0.9).build()
pattern_order = r'[0-9]'

if __name__ == "__main__":
    #SCRAPING START
    html = urlopen(url).read()
    html_female = urlopen(url_female).read()
    soup = BeautifulSoup(html, features="html.parser")
    soup2 = BeautifulSoup(html_female, features="html.parser")


    while len(titrip)<48000:
        title= soup.find_all("span", {
                "class": "title"})
        get_title(soup, titrip)
        rat = soup.find_all("div", {
            "class": "value"})
        get_rating(soup, rating)
        view = soup.find_all("div", {
        "class": "value"})
        get_views(soup, views)
        times = soup.find_all("var", {
            "class": "added"})
        get_time(soup, time)

        title_fem = soup2.find_all("span", {
            "class": "title"})
        get_title(soup2, titrip_fem)
        rat_fem = soup2.find_all("div", {
            "class": "value"})
        get_rating(soup2, rating_fem)
        view_fem = soup2.find_all("div", {
            "class": "value"})
        get_views(soup2, views_fem)
        times_fem = soup2.find_all("var", {
            "class": "added"})
        get_time(soup2, time_fem)
        if len(titrip)<48000:
            i+=1
            i_link = str(i)
            prov_url = url + link_next_page + i_link
            print(prov_url)
            html = urlopen(prov_url).read()
            soup = BeautifulSoup(html, features="html.parser")
    data = {
        'title_videos': titrip
    }

    data_fem = {
        'title_female_videos': titrip_fem
    }
    df_title = pd.DataFrame(data)
    df_title_fem = pd.DataFrame(data_fem)
    df_title.to_csv("title_videos.csv")
    df_title_fem.to_csv("title_videos_fem.csv")


    #Topic Modelling
    datasets = [(df_title["title_videos"],"General Videos"), (df_title_fem["title_female_videos"], "Female Videos")]
    no_features = 1000
    tf_vectorizer = CountVectorizer(max_df=0.95,
                                    min_df=2,
                                    max_features=no_features,
                                    stop_words='english')
    for dataset in datasets:
        tf = tf_vectorizer.fit_transform(dataset[0])
        tf_feature_names = tf_vectorizer.get_feature_names_out()
        lda = LatentDirichletAllocation(random_state=0)


        searchParams = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}  # n components 10,15,20,25,30
        model = GridSearchCV(lda,
                             param_grid=searchParams,
                             verbose=3,
                             n_jobs=-1)

        model.fit(tf)
        best_lda_model = model.best_estimator_
        print(model)
        print("Best Log Likelihood Score: ", model.best_score_)
        print("Best Model's Params: ", model.best_params_)
        print("Model Perplexity: ", best_lda_model.perplexity(tf))

        n_topics = [10, 15, 20, 25, 30]
        log_likelyhoods_5 = [round(model.cv_results_['mean_test_score'][index])
                             for index, gscore in enumerate(model.cv_results_['params']) if gscore['learning_decay'] == 0.5]
        log_likelyhoods_7 = [round(model.cv_results_['mean_test_score'][index])
                             for index, gscore in enumerate(model.cv_results_['params']) if gscore['learning_decay'] == 0.7]
        log_likelyhoods_9 = [round(model.cv_results_['mean_test_score'][index])
                             for index, gscore in enumerate(model.cv_results_['params']) if gscore['learning_decay'] == 0.9]

        # Show graph
        plt.figure(figsize=(12, 8))
        plt.plot(n_topics, log_likelyhoods_5, label='0.5')
        plt.plot(n_topics, log_likelyhoods_7, label='0.7')
        plt.plot(n_topics, log_likelyhoods_9, label='0.9')
        plt.title("Choosing Optimal LDA Model")
        plt.xlabel("Num Topics")
        plt.ylabel("Log Likelyhood Scores")
        plt.legend(title='Learning decay', loc='best')
        plt.show()

        panel = pyLDAvis.sklearn.prepare(lda_model=best_lda_model, dtm=tf, vectorizer=tf_vectorizer)
        # panel

        name = dataset[1].lower()
        pyLDAvis.save_html(panel, f'LDA_panel_{name}.html')


