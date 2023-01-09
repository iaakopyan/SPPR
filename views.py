from django.shortcuts import render
from django.views.generic import ListView, CreateView
# from .models import Student, Department, Employee
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse
# from .forms import StudentForm, DepartmentForm, EmployeeForm

#
# class StudentView(ListView):
#     model = Student
#
#     def get(self, request, **kwargs):
#         students = Student.objects.all()
#         return HttpResponse(students)
#
#
# class StudentProcessing(LoginRequiredMixin, CreateView):
#     model = Student
#     form_class = StudentForm
#
#     def form_valid(self, form: StudentForm)-> HttpResponse:
#         return super().form_valid(form)  # отвечает за валидацию
#
#
# class DepartmentView(ListView):
#     model = Department
#
#     def get(self, request, **kwargs):
#         departments = Department.objects.all()
#         return HttpResponse(departments)
#
#
# class DepartmentProcessing(LoginRequiredMixin, CreateView):
#     model = Department
#     form_class = DepartmentForm
#
#     def form_valid(self, form: DepartmentForm)-> HttpResponse:
#         return super().form_valid(form)  # отвечает за валидацию
#
#
# class EmployeeView(ListView):
#     model = Department
#
#     def get(self, request, **kwargs):
#         employees = Department.objects.all()
#         return HttpResponse(employees)
#
#
# class EmployeeProcessing(LoginRequiredMixin, CreateView):
#     model = Employee
#     form_class = EmployeeForm
#
#     def form_valid(self, form: EmployeeForm)-> HttpResponse:
#         return super().form_valid(form)  # отвечает за валидацию
from surprise.model_selection import GridSearchCV

from .forms import UnitForm
from .models import Unit

import pickle
import numpy as np
import pandas as pd
import os
import seaborn as sns
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import train_test_split

import difflib
import random

user_rating_path = 'media/BX-Book-Ratings.csv'
book_info_path = 'media/BX_Books.csv'
data = pd.read_csv(user_rating_path, sep=',', encoding='latin-1', error_bad_lines=False)
data = data.loc[data['BookRating'] != 0]
book_info_data = pd.read_csv(book_info_path, sep=',', encoding='latin-1', error_bad_lines=False)
data = pd.merge(data, book_info_data[['ISBN', 'Book-Title']], on='ISBN')
data = data[['UserID', 'ISBN', 'BookRating', 'Book-Title']]
data = data[['UserID', 'ISBN', 'BookRating', 'Book-Title']]

reader = Reader(rating_scale=(0, 10))
rating_data = Dataset.load_from_df(data[['UserID', 'ISBN', 'BookRating']], reader)

# svd = pickle.load(open('media/test_model.sav', 'rb'))


def get_book_id(book_name, data):
    book_names = list(data['Book-Title'].values)
    closest_names = difflib.get_close_matches(book_name, book_names)
    book_id = data[data['Book-Title'] == closest_names[0]]['ISBN'].iloc[0]
    print(book_id)
    return book_id


def predict_rating(user_id, book_name, data, model=SVD):
    book_id = get_book_id(book_name, data)
    reader = Reader(rating_scale=(0.5, 10))
    # testset_model_surprise = Dataset.load_from_df(data[['UserID', 'ISBN', 'BookRating']], reader).build_full_trainset()
    # trainset_model_surprise = Dataset.load_from_df(data[['UserID', 'ISBN', 'BookRating']], reader).build_full_trainset()
    # m = SVD()
    # testset_surprise = testset_model_surprise.build_testset()
    # model_normal_predictor_surprise = m.fit(trainset_model_surprise)
    # # predictions = model_normal_predictor_surprise.test(testset_surprise)
    # pickle.dump(model_normal_predictor_surprise, open('test_model.sav', 'wb'))

    trainset, testset = train_test_split(data, test_size=0.25)
    algo = SVD(n_factors=25, n_epochs=15, lr_all=0.003, reg_all=0.1)
    predictions = algo.fit(trainset).test(testset)
    pickle.dump(predictions, open('test_model4.sav', 'wb'))

    estimated_ratings = gs.predict(uid=user_id, iid=book_id)
    return estimated_ratings.est


def ten_users():
    users = []
    for i in range(10):
        random_user_id = np.random.choice(list(np.unique(data['UserID'].values)))
        users.append(
            str(str(random_user_id) + " : " + str(pd.Series(data.loc[np.where(data.UserID == random_user_id)]
                                                   ['Book-Title'].values))))
    return users


def recommend_books(user_id, data=data, model=SVD, threshold=0.1):
    recommended_books = {}
    unique_book_names = list(np.unique(data['Book-Title'].values))
    random.shuffle(unique_book_names)
    print(len(unique_book_names), 'jdnvkjbdfkj')
    for book_name in unique_book_names:
        rating = predict_rating(user_id=user_id, book_name=book_name, data=data, model=SVD)
        if rating > threshold:
            recommended_books[book_name] = np.round(rating, 2)

    print("Generation recomandation of books for user ID {} : ".format(user_id))

    book_names = np.array(list(recommended_books.keys())).reshape(-1, 1)
    book_ratings = np.array(list(recommended_books.values())).reshape(-1, 1)
    results = np.concatenate((book_names, book_ratings), axis=1)
    results_df = pd.DataFrame(results, columns=['Books', 'Rating (0-10)']).sort_values(by='Rating (0-10)',
                                                                                            ascending=False)
    res = results_df.reset_index().drop('index', axis=1)
    return res


def index(request):
    return render(request, 'index.html')


def recom(request):
    work = Unit.objects.order_by('-user_id')
    if work:
        for n in work:
            if n.recomendations == "NULL":
                n.recomendations = recommend_books(user_id=n.user_id)
                n.save()
    return render(request, 'recom.html', {'work': work})


def post(request):
    ex = ten_users()
    error = ''
    if request.method == 'POST':
        form = UnitForm(request.POST)

        if form.is_valid():
            form.save()
        else:
            error = 'Ошибка заполнения'

    form = UnitForm()
    data = {
        'form': form,
        'error': error,
        'ex': ex
    }
    return render(request, 'post.html', data)
