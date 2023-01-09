# from django import forms

# class StudentForm(forms.Form):
#     name = forms.CharField()
#     id_group = forms.IntegerField()
#     department = forms.IntegerField()
#     age = forms.IntegerField()
#
# class DepartmentForm(forms.Form):
#         id = forms.IntegerField()
#         Name = forms.CharField()
#
# class EmployeeForm(forms.Form):
#     id = forms.IntegerField()
#     Name = forms.CharField()
#     department = forms.IntegerField()

from .models import Unit
from django.forms import ModelForm, TextInput


class UnitForm(ModelForm):
    class Meta:
        model = Unit
        fields = ['user_id']

        widgets = {
            "user_id": TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'user_id'
            }),
        }
