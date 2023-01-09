from django.db import models


# class Department(models.Model):
#     id = models.IntegerField(primary_key=True, unique=True)
#     Name = models.CharField(max_length=128, blank=True, null=True)
#
#
# class Employee(models.Model):
#     id = models.IntegerField(primary_key=True, unique=True)
#     Name = models.CharField(max_length=128, blank=True, null=True)
#     department = models.ForeignKey(Department, on_delete=models.CASCADE)
#
#
# class Student(models.Model):
#     id = models.IntegerField(primary_key=True, unique = True)
#     Name = models.CharField(max_length = 128, blank=True, null=True)
#     id_group = models.IntegerField(blank=True, null=True)
#     department = models.ForeignKey(Department, on_delete=models.CASCADE)
#     age = models.IntegerField(blank=True, null=True)

class Unit(models.Model):
    user_id = models.TextField('user_id', default=0)
    recomendations = models.TextField('Рекомендация', default="NULL")

    def __str__(self):
        return self.title

    class Meta:
        verbose_name = 'Объект'
        verbose_name_plural = 'Объекты'
