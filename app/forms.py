from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField, BooleanField, SelectField
from wtforms.fields.core import IntegerField
from wtforms.validators import DataRequired
from markupsafe import Markup
class SubmitForm(FlaskForm):
    DGN = SelectField('Please enter the disease your diagnosed wth from the following options: ', choices=[ "DGN3", "DGN2", "DGN4", "DGN6", "DGN5", "DGN8", "DGN1"], validators=[DataRequired()])
    PRE4 = FloatField('Please enter your forced vital capacity:', validators=[DataRequired()])
    PRE5 = FloatField('Please enter the value of you FEV1:', validators=[DataRequired()])
    PRE6 = SelectField('Please enter the disease your Performance Status from the following options:', choices = ["PRZ2", "PRZ1", "PRZ0"], validators=[DataRequired()])
    PRE7 = BooleanField("Any sorts of pain.")
    PRE8 = BooleanField("Haemoptysis")
    PRE9 = BooleanField("Dyspnoea")
    PRE10 = BooleanField("Cough ")
    PRE11 = BooleanField("Any weakness")
    PRE14 = SelectField("Please enter the size of the tumour from the following options:",choices=["OC11", "OC12", "OC13", "OC14"], validators=[DataRequired()])
    PRE17 = BooleanField("Any diabetes mellitus(DM)")
    PRE19 = BooleanField("Any MI (Myocardial Infarction)")
    PRE25 = BooleanField("Any peripheral arterial diseases")
    PRE30 = BooleanField("Habit of smoking")
    PRE32 = BooleanField("Asthma")
    AGE = IntegerField("Please enter your age", validators=[DataRequired()])
    submit = SubmitField("Submit")