from flask import Flask, render_template
from has import has
from haf import haf

app = Flask(__name__,static_folder='static')
app.register_blueprint(has,url_prefix='/has')
app.register_blueprint(haf,url_prefix='/haf')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/usecase')
def usecase():
    return render_template('usecase.html')

@app.route('/try')
def try_it():
    return render_template('try.html')

@app.route('/patient_therapist/')
def patient_therapist_home():
    return render_template('patient_home.html')

@app.route('/security/')
def security_home():
    return render_template('security_home.html')

@app.route('/customer/')
def customer_home():
    return render_template('customer_home.html')

if __name__ == '__main__':
    app.run(debug=True)