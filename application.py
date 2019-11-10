from flask import Flask

app = Flask(__name__)

@app.route("/hello")
def sayhello():
    return "Hello World!"

@app.route("/prabodhan")
def sayhellotoprabodhan():
    return "Hello Prabodhan!"

if __name__ == "__main__":
    app.run('localhost',5000)

