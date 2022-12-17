from collections import defaultdict

import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from surprise import *
from surprise import accuracy
from surprise.model_selection import GridSearchCV, train_test_split
from sklearn.cluster import KMeans

books_ratings = pd.read_csv('BX-Book-Ratings.csv', sep=',', encoding='latin-1', error_bad_lines=False)
books = pd.read_csv('BX_Books.csv', sep=',', encoding='latin-1', error_bad_lines=False)
users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
# print(books.head())#ok
# print(books_ratings.head())#ok
reader = Reader(line_format='user item rating', sep="\t")
reader = Reader(rating_scale=(0.5, 10.0))
df = Dataset.load_from_df(books_ratings, reader)


def configure_plotly_browser_state():
    import IPython
    display(IPython.core.display.HTML('''


        '''))


configure_plotly_browser_state()
import plotly.graph_objs as go

data = books_ratings["BookRating"].value_counts().sort_index(ascending=False)
trace = go.Bar(x=data.index,
               text=['{:.1f} %'.format(val) for val in (data.values / books_ratings.shape[0] * 100)],
               textposition='auto',
               textfont=dict(color='#000000'),
               y=data.values,
               )
# Create layout
layout = dict(title='Distribution Of {} ratings'.format(books_ratings.shape[0]),
              xaxis=dict(title='Rating'),
              yaxis=dict(title='Count'))
# Create plot
fig = go.Figure(data=[trace], layout=layout)
# fig.show() #ok

data2 = books_ratings.groupby('ISBN')['BookRating'].count().clip(upper=50)
# Create trace
trace2 = go.Histogram(x=data2.values,
                      name='Ratings',
                      xbins=dict(start=0,
                                 end=50,
                                 size=2))
# Create layout
layout2 = go.Layout(title='Distribution Of Number of Ratings Per Item (Clipped at 50)',
                    xaxis=dict(title='Number of Ratings Per Item'),
                    yaxis=dict(title='Count'),
                    bargap=0.2)

# Create plot
fig2 = go.Figure(data=[trace2], layout=layout2)

# fig2.show() #ok
# print(data2.head()) #ok
# print(books_ratings.shape)
books_ratings = books_ratings[books_ratings.ISBN.isin(books.ISBN)]
# print(books_ratings.shape)
ratings_explicit = books_ratings[books_ratings.BookRating != 0]
ratings_implicit = books_ratings[books_ratings.BookRating == 0]
import seaborn as sns
import matplotlib.pyplot as plt

counts1 = pd.value_counts(ratings_explicit['UserID'])
ratings_explicit_new = ratings_explicit[ratings_explicit['UserID'].isin(counts1[counts1 >= 500].index)]
data = Dataset.load_from_df(ratings_explicit[['UserID', 'ISBN', 'BookRating']], reader)
# print(data.df.head()) #ok
trainset, testset = train_test_split(data, test_size=0.25)
algo = SVD(n_factors=40, n_epochs=25, lr_all=0.008, reg_all=0.08)
predictions = algo.fit(trainset).test(testset)
print(accuracy.rmse(predictions))
svd_model = SVD(n_factors=5)
svd_model.fit(trainset)

test_pred = svd_model.test(testset)


# print(test_pred) #ok
def get_Iu(uid):
    """
    args:
      uid: the id of the user
    returns:
      the number of items rated by the user
    """
    try:
        return len(trainset.ur[trainset.to_inner_uid(uid)])
    except ValueError:  # user was not part of the trainset
        return 0


def get_Ui(iid):
    """
    args:
      iid: the raw id of the item
    returns:
      the number of users that have rated the item.
    """
    try:
        return len(trainset.ir[trainset.to_inner_iid(iid)])
    except ValueError:
        return 0


df_predictions = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])
df_predictions['Iu'] = df_predictions.uid.apply(get_Iu)
df_predictions['Ui'] = df_predictions.iid.apply(get_Ui)
df_predictions['err'] = abs(df_predictions.est - df_predictions.rui)
# print(df_predictions.head()) #ok
best_predictions = df_predictions.sort_values(by='err')[:10]
worst_predictions = df_predictions.sort_values(by='err')[-10:]
temp = books_ratings.loc[books_ratings['ISBN'] == 3996]['BookRating']
configure_plotly_browser_state()

# Create trace
trace3 = go.Histogram(x=temp.values,
                      name='BookRating',
                      xbins=dict(start=0,
                                 end=5, size=.3))
# Create layout
layout3 = go.Layout(title='Number of ratings item 3996 has received',
                    xaxis=dict(title='Number of Ratings Per Item'),
                    yaxis=dict(title='Count'),
                    bargap=0.2)
# Create plot
fig3 = go.Figure(data=[trace3], layout=layout3)
# (fig3.show()) #NOT ok

final = []

for threshold in np.arange(0, 5.5, 0.5):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    temp = []

    for uid, _, true_r, est, _ in predictions:
        if (true_r >= threshold):
            if (est >= threshold):
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if (est >= threshold):
                fp = fp + 1
            else:
                tn = tn + 1

        if tp == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)

    temp = [threshold, tp, fp, tn, fn, precision, recall, f1]
    final.append(temp)

results = pd.DataFrame(final)
results.rename(columns={0: 'threshold', 1: 'tp', 2: 'fp', 3: 'tn', 4: 'fn', 5: 'Precision', 6: 'Recall', 7: 'F1'},
               inplace=True)


# print(results) #ok

def precision_recall_at_k(predictions, k, threshold):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls


results = []
for i in range(2, 11):
    precisions, recalls = precision_recall_at_k(predictions, k=i, threshold=2.5)

    # Precision and recall can then be averaged over all users
    prec = sum(prec for prec in precisions.values()) / len(precisions)
    rec = sum(rec for rec in recalls.values()) / len(recalls)
    results.append({'K': i, 'Precision': prec, 'Recall': rec})

# print(results) #ok
Rec = []
Precision = []
Recall = []
for i in range(0, 9):
    Rec.append(results[i]['K'])
    Precision.append(results[i]['Precision'])
    Recall.append(results[i]['Recall'])

from matplotlib import pyplot as plt

plt.plot(Rec, Precision)
plt.xlabel('# of Recommendations')
plt.ylabel('Precision')
plt2 = plt.twinx()
plt2.plot(Rec, Recall, 'r')
plt.ylabel('Recall')
for tl in plt2.get_yticklabels():
    tl.set_color('r')
# plt.show()
trainset = data.build_full_trainset()  # Build on entire data set
algo = SVD(n_factors=35, n_epochs=25, lr_all=0.008, reg_all=0.08)
algo.fit(trainset)

# Predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()

# Predicting the ratings for testset
predictions = algo.test(testset)


def get_all_predictions(predictions):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)

    return top_n


all_pred = get_all_predictions(predictions)
n = 4

for uid, user_ratings in all_pred.items():
    user_ratings.sort(key=lambda x: x[1], reverse=True)
    all_pred[uid] = user_ratings[:n]
tmp = pd.DataFrame.from_dict(all_pred)
tmp_transpose = tmp.transpose()
def get_predictions(UserID):
    results = tmp_transpose.loc[UserID]
    return results

results = get_predictions(2)
#print(results)

recommended_book_ids=[]
for x in range(0, n):
    recommended_book_ids.append(results[x][0])
print(recommended_book_ids)


recommended_books = books[books['movieId'].isin(recommended_book_ids)]
print(recommended_books)

temp = books_ratings[books_ratings['UserId'] == 2].sort_values("BookRating", ascending = False)
print(temp.head())

history_book_ids = temp['ISBN']
user_history = books[books[''].isin(history_book_ids)]
print(user_history[:n])
print(recommended_book_ids)