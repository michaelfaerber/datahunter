"""The views module connects the GUI with functionality."""

import datetime
import string
import pickle
import re
import fasttext
from io import BytesIO
import pandas as pd
import numpy as np
from matplotlib import pyplot
from django.shortcuts import render
from django.core.paginator import Paginator
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from rank_bm25 import BM25Plus
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib
matplotlib.use('Agg')
from .filters import DatasetFilter
from .models import Dataset, Link, Paper


def preprocess(input_string):
    """
    Natural language preprocessing: only lower-case, apostrophes, punctuation and numbers are
    removed, stopwords are erased and single characters are removed, word stemming is applied
    input: input_string,
    output: preprocessed_string
    """
    input_string = input_string.lower()
    input_string = input_string.replace("'", "")
    input_string = input_string.translate(str.maketrans("", "", string.punctuation))
    input_string = input_string.translate(str.maketrans("", "", string.digits))
    tokens = word_tokenize(input_string)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    result_string = ""
    for word in tokens:
        if word not in stop_words:
            if len(word) > 1:
                transformed_word = stemmer.stem(word)
                result_string = result_string + transformed_word + " "
    preprocessed_string = result_string.strip()
    return preprocessed_string

def tfidf(document_collection):
    """
    Computes tfidf for each document in the document_collectionlection that is input
    input: list of documents document_collection,
    output: list of tfidfs scores for documents tf_document_collection
    """
    transformer = TfidfTransformer()
    vocab = pickle.load(open("/vol3/ann-kathrin/tfidf_vocab.pkl", "rb"))
    loaded_vec = TfidfVectorizer(decode_error="replace", min_df=1, vocabulary=vocab)
    tf_document_collection = transformer.fit_transform(loaded_vec.fit_transform(
        np.array(document_collection)))
    return tf_document_collection

def home(request):
    """Method that opens or redirects to home.html."""
    return render(request, 'home.html')

def no_link(request):
    """Method that redirects to no_link.html."""
    return render(request, 'no_link.html')

def possible_queries(request):
    f = open('/vol3/ann-kathrin/4643Ann-Kathrin_Leisinger/Daten/Abstracts/Samples_Queries_Abstracts.txt', 'r')
    abstracts = []
    for line in f:
        abstracts.append(line)
    f.close()
    context = {'result' : abstracts }
    return render(request, "sample_queries.html", context)

def sorting_datestrings(datelist):
    """
    This method helps to sort a list of dates from DSKG.csv by transforming year, month and day
    to integers and returning them as keys
    """
    splitup = datelist.split('-')
    year = int(splitup[0])
    month = int(splitup[1])
    day_hour = splitup[2].split("T")
    day = int(day_hour[0])
    return year, month, day

def remove_html(text):
    """
    This method removes html tags from a text to clean it up
    """
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', text)
    return cleantext

def dataset_prediction(query):
    """
    In this function the query is preprocessed and tfidf is computed and the pretrained SVM model
    predicts relevant dataset for this query. Furhtermore, the BM25 ranking function is applied if
    possible.
    input: query
    output: indices of predicted datasets, array with ranking scores for datasets
    """
    ### For tfidf model
    #preprocessed_query = tfidf([preprocess(query)])
    #model = pickle.load(open(
    #    '/vol3/ann-kathrin/svm_tfidf.sav',
    #    'rb'))
    #label_encoder = pickle.load(open(
    #    '/vol3/ann-kathrin/label_encoder_tfidf.sav',
    #    'rb'))
    ###
    preprocessed_query = preprocess(query)
    model = fasttext.load_model('/vol3/ann-kathrin/model_fasttext_classification_rawtext.bin')
    prediction = model.predict(preprocessed_query, k=5)
    prediction_clean = str(prediction[0]).replace("__label__", "").replace("(", "").replace(")", "").replace("'", "")
    ### For tfidf model
    #class_indices = [i for i, x in enumerate(prediction[0]) if x == 1]
    #prediction_transformed = label_encoder.inverse_transform(prediction)
    #prediction_clean = str(prediction_transformed).replace("[", "").replace("]", "")
    #prediction_clean = prediction_clean.replace("(", "").replace(")", "").replace("'", "")
    #prediction_clean = prediction_clean.replace(" '", "").replace("' ", "").strip()
    ###
    prediction_list = prediction_clean.split(",")
    dataset_file = pd.read_csv(
        "/vol3/ann-kathrin/DSKG_FINAL_TABELLENFORM.csv")
    with open("log.txt", "a") as file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write("{}\t{}\t{}\n".format(timestamp, query, prediction_list))
        file.flush()
    prediction_indices = []
    selected_datasets = []
    for predicted_dataset in prediction_list:
        if predicted_dataset is not '' and predicted_dataset is not None:
            index = dataset_file[dataset_file['dataset'] == str(predicted_dataset).strip()].index.values
            for i in index:
                prediction_indices.append(int(i))
                index_description_tuple = (int(i), str(dataset_file['description'][int(i)]))
                selected_datasets.append(index_description_tuple)
    ### For tfidf model
    #confidence_scores = model.decision_function(preprocessed_query)
    #scoring_list = []
    #for i in range(len(selected_datasets)):
    #    dataset_score_triple = (selected_datasets[i][0], selected_datasets[i][1], abs(confidence_scores[0][int(class_indices[i])]))
    #    scoring_list.append(dataset_score_triple)
    #print(str(prediction[1]).replace("[", "").replace("]", "").split(" "))
    ###
    scoring_list = []
    for i in range(len(selected_datasets)):
         dataset_score_triple = (selected_datasets[i][0], selected_datasets[i][1], prediction[1][i])
         scoring_list.append(dataset_score_triple)
    ranking_array = sorted(scoring_list, key=lambda l: l[2], reverse=False)
    return prediction_indices, ranking_array

def recommendation(request):
    """Filters list of Dataset objects and modates this list based on user's filter inputs."""
    Dataset.objects.all().delete()
    Link.objects.all().delete()
    context = {}
    query_text = request.GET.get('querytext', "")
    indices, ranking_array = dataset_prediction(query_text)
    dataframe = pd.read_csv(
        "/vol3/ann-kathrin/DSKG_FINAL_TABELLENFORM.csv")
    references_frame = pd.read_csv(
        "/vol3/ann-kathrin/DSKG_MAKG.csv")
    with open("/vol3/ann-kathrin/4643Ann-Kathrin_Leisinger/Daten/PaperTitles.txt", 'r') as titles_frame:
        titles = {int(line.split("\t")[0]): line.split("\t")[1] for line in titles_frame}
    for index in indices:
        if not pd.isnull(dataframe['title'][index]):
            dataset_title = dataframe['title'][index]
        else:
            dataset_title = ""
        if not pd.isnull(dataframe['description'][index]):
            dataset_description = remove_html(dataframe['description'][index].strip())
        else:
            dataset_description = ""
        dataset_topic = ""
        if not pd.isnull(dataframe['keyword'][index]):
            keyword = str(dataframe['keyword'][index]).strip()
            keyword = keyword.translate(str.maketrans("", "", string.digits))
            #keyword = keyword.translate(str.maketrans("", "", string.punctuation))
            dataset_topic = dataset_topic + keyword
        if not pd.isnull(dataframe['alternative'][index]):
            alternative = str(dataframe['alternative'][index]).strip()
            alternative = alternative.translate(str.maketrans("", "", string.digits))
            #alternative = alternative.translate(str.maketrans("", "", string.punctuation))
            if dataset_topic != "":
                dataset_topic = dataset_topic + ", "
            dataset_topic = dataset_topic + alternative
        if not pd.isnull(dataframe['theme'][index]):
            theme = str(dataframe['theme'][index]).strip()
            #TODO get label of theme and create link
            #theme = theme.translate(str.maketrans("", "", string.digits))
            #theme = theme.translate(str.maketrans("", "", string.punctuation))
            #if dataset_topic != "":
            #    dataset_topic = dataset_topic + ", "
            #dataset_topic = dataset_topic + theme
        dataset_url_list = []
        if not pd.isnull(dataframe['landingPage'][index]):
            landingPage = str(dataframe['landingPage'][index]).split(",")
            for url in landingPage:
                if url not in dataset_url_list:
                    dataset_url_list.append(url)
        if not pd.isnull(dataframe['accessURL'][index]):
            accessURL = str(dataframe['accessURL'][index]).split(",")
            for url in accessURL:
                if url not in dataset_url_list:
                    dataset_url_list.append(url)
        print(dataset_url_list)
        if not pd.isnull(dataframe['identifier'][index]):
            dataset_identifier = dataframe['identifier'][index]
        else:
            dataset_identifier = ""
        if not pd.isnull(dataframe['creatorName'][index]):
            dataset_creator = dataframe['creatorName'][index]
        else:
            dataset_creator = ""
        if not pd.isnull(dataframe['publisherName'][index]):
            dataset_publisher = dataframe['publisherName'][index]
        else:
            dataset_publisher = ""
        if not pd.isnull(dataframe['contributorName'][index]):
            dataset_contributor = dataframe['contributorName'][index]
        else:
            dataset_contributor = ""
        if not pd.isnull(dataframe['accessRights'][index]):
            dataset_access_right = dataframe['accessRights'][index]
        else:
            dataset_access_right = ""
        if not pd.isnull(dataframe['byteSize'][index]):
            dataset_size = float(dataframe['byteSize'][index])/1000
        else:
            dataset_size = None
        if not pd.isnull(dataframe['format'][index]):
            dataset_data_format = dataframe['format'][index]
        else:
            dataset_data_format = ""
        if not pd.isnull(dataframe['source'][index]):
            if "http://www.wikidata.org/entity" in str(dataframe['source'][index]):
                dataset_source = "Wikidata"
            elif "https://zenodo.org/record" in str(dataframe['source'][index]):
                dataset_source = "OpenAire"
            else:
                dataset_source = str(dataframe['source'][index])
        else:
            dataset_source = ""
        if not pd.isnull(dataframe['issued'][index]):
            datelist = str(dataframe['issued'][index]).split(",")
            datelist.sort(key=sorting_datestrings)
            date = datelist[0].split("-")
            issued_year = int(date[0])
            issued_month = int(date[1])
            issued_day_hour = date[2].split("T")
            issued_day = int(issued_day_hour[0])
            dataset_issued_date = datetime.date(issued_year, issued_month, issued_day)
        else:
            dataset_issued_date = None
        if not pd.isnull(dataframe['modified'][index]):
            datelist = str(dataframe['modified'][index]).split(",")
            datelist.sort(key=sorting_datestrings)
            date = datelist[0].split("-")
            modified_year = int(date[0])
            modified_month = int(date[1])
            modified_day_hour = date[2].split("T")
            modified_day = int(modified_day_hour[0])
            dataset_modified_date = datetime.date(modified_year, modified_month, modified_day)
        else:
            dataset_modified_date = None
        if not pd.isnull(dataframe['language'][index]):
            dataset_language = dataframe['language'][index]
        else:
            dataset_language = None
        ranking_array_object = [x for x in ranking_array if x[0] == index]
        ranking_array_index = ranking_array.index(ranking_array_object[0])
        if not ranking_array_index is None:
            dataset_ranking_score = ranking_array[int(ranking_array_index)][2]
        else:
            dataset_ranking_score = None
        dataset_referenced_papers_list = str(references_frame['isReferencedBy'][index]).split(",")
        dataset_referenced_papers = []
        dataset_referenced_papers_string = ""

        papers_max = 5
        papers_count = 0
        for paper in dataset_referenced_papers_list:
            if papers_count >= papers_max:
                break
            papers_count += 1
            referenced_paper = {}
            paperid = str(paper).replace("http://ma-graph.org/entity/", "").strip()
            text = str(paperid)
            referenced_paper["id"] = text
            #referenced_paper["url"] = paper # URL to RDF entity
            referenced_paper["url"] = "https://academic.microsoft.com/paper/" + text # URL to human-readable MAKG site
            try:
                title = titles.get(int(paperid))
                text = text + " (" + str(title) + ")"
                if title is not None and title != "":
                    referenced_paper["title"] = title
            except:
                pass
            text = text.replace("(None)", "")
            if paper == dataset_referenced_papers_list[0]:
                dataset_referenced_papers_string = dataset_referenced_papers_string + " " + text
            else:
                dataset_referenced_papers_string = dataset_referenced_papers_string + ", " + text

            dataset_referenced_papers.append(referenced_paper)

        dataset = Dataset.objects.create(title=dataset_title, description=dataset_description,
                                         topic=dataset_topic,
                                         identifier=dataset_identifier,
                                         creator=dataset_creator, publisher=dataset_publisher,
                                         contributor=dataset_contributor,
                                         access_right=dataset_access_right,
                                         size=dataset_size, data_format=dataset_data_format,
                                         source=dataset_source, issued_date=dataset_issued_date,
                                         modified_date=dataset_modified_date,
                                         language=dataset_language,
                                         ranking_score=dataset_ranking_score,
                                         referenced_papers_string=dataset_referenced_papers_string)
        dataset.save()

        # URLs
        if len(dataset_url_list) > 1:
            dataset_url_list = [dataset_url_list[0]]
        for url in dataset_url_list:
            link = Link.objects.create(url=url)
            link.save()
            dataset.link.add(link)

        # Papers
        for paper in dataset_referenced_papers:
            if "title" in paper:
                paper = Paper.objects.create(url=paper["url"], paperid=paper["id"], title=paper["title"])
            else:
                paper = Paper.objects.create(url=paper["url"], paperid=paper["id"])
            paper.save()
            dataset.referenced_papers.add(paper)




    col = []
    for dataset in Dataset.objects.all():
        title_description_tuple = (dataset.title, dataset.description)
        col.append(title_description_tuple)
    if len(col) > 0:
        context['recommendation_empty'] = False
    else:
        context['recommendation_empty'] = True
    filtered_datasets = DatasetFilter(request.GET,
                                      queryset=Dataset.objects.all().order_by('ranking_score'))
    context['filtered_datasets'] = filtered_datasets
    paginated_filtered_datasets = Paginator(filtered_datasets.qs, 4)
    page_number = request.GET.get('page')
    dataset_page_object = paginated_filtered_datasets.get_page(page_number)
    context['dataset_page_object'] = dataset_page_object
    return render(request, 'results.html', context=context)
