from flask import Flask, render_template

app = Flask(__name__,static_folder='static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/usecase')
def usecase():
    return render_template('usecase.html')

@app.route('/try')
def try_it():
    return render_template('try.html')

@app.route('/patient_therapist/home')
def patient_therapist_home():
    return render_template('patient_home.html')

@app.route('/security/home')
def security_home():
    return render_template('security_home.html')

@app.route('/customer/home')
def customer_home():
    return render_template('customer_home.html')

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)