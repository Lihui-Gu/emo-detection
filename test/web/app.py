import pandas as pd
from flask import Flask, render_template

app = Flask(__name__)
def generate_datas():
    datas = {'专注':10, '不专注': 8}
    return datas

@app.route('/')
def index():
    datas = generate_datas()
    return render_template('index.html', datas=datas)

if __name__ == '__main__':
    datas = generate_datas()
    for data in datas:
        print(datas[data])
    app.run(host='0.0.0.0', debug=True, port=8080)


