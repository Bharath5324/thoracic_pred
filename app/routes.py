from flask.helpers import get_flashed_messages
from app import app
from flask import render_template, url_for, redirect, flash
from app.forms import SubmitForm
from app import predictor

@app.route('/result', methods=['GET'])
def result():
    ans = get_flashed_messages()
    print(ans)
    if len(ans)>0:
        if ans[0] == '0':
            x = "Nah dwag, you're fine!!"
        elif ans[0] == '1':
            x = "Sorry bud, I don't reccomend you go through with the suregery!"
    else:
        return(redirect(url_for('index')))
    return render_template('result.html', x=x, title="Result")

@app.route('/', methods=['GET', 'POST'])
def index():
    form = SubmitForm()
    if form.validate_on_submit():
        print("=============[ Entry Confirmation ]=============")
        x = form.data
        x.pop('csrf_token')
        x.pop('submit')
        flash(str(predictor.predict(x)))
        print(result)
        return redirect(url_for('result'))
    return render_template('form.html', form=form, title="Form")
