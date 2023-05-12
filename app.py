from flask import Flask, render_template, redirect, url_for, request, session
from User1_5.signature_user1_5 import signature_verification
from User6_10.signature_user6_10 import signature_verification2

app = Flask(__name__)
app.secret_key = 'my_secret_key'

users = [
    {'username': 'user1', 'password': 'password1'},
    {'username': 'user2', 'password': 'password2'},
]

last_verification_result = None
last_verification_result2 = None

@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        for user in users:
            if user['username'] == username and user['password'] == password:
                session['username'] = username
                return render_template('index.html')
        return 'Invalid username or password'
    else:
        return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        for user in users:
            if user['username'] == username:
                return 'Username already taken'
        users.append({'username': username, 'password': password})
        session['username'] = username
        return redirect(url_for('login'))
    else:
        return render_template('signup.html')

@app.route("/reference", methods=["POST", "GET"])
def reference():
    if request.method == "POST":
        username = request.form["inputUserNumber"]
        signature_path = request.form["signature_path"]
        verification_result = signature_verification(username, signature_path)

        global last_verification_result
        last_verification_result = None
        last_verification_result = verification_result

        return redirect(url_for("result", usr=username))
    else:
        return render_template("reference.html")

@app.route("/result")
@app.route("/result/<usr>")

def result(usr=None):
    if usr is None:
        return "No user provided or no verification result", 400

    global last_verification_result
    verification_result = last_verification_result

    print(verification_result)
    print(usr)

    verification_result_dict = verification_result

    print("--------------------------------")
    print(verification_result_dict)

    return render_template("result.html", **verification_result_dict)


@app.route("/reference2", methods=["POST", "GET"])
def reference2():
    if request.method == "POST":
        username = request.form["inputUserNumber"]
        signature_path = request.form["signature_path"]
        verification_result = signature_verification2(username, signature_path)

        global last_verification_result2
        last_verification_result2 = None
        last_verification_result2 = verification_result

        return redirect(url_for("result2", usr=username))
    else:
        return render_template("reference2.html")

@app.route("/result2")
@app.route("/result2/<usr>")

def result2(usr=None):
    if usr is None:
        return "No user provided or no verification result", 400

    global last_verification_result2
    verification_result = last_verification_result2

    print(verification_result)
    print(usr)

    verification_result_dict = verification_result

    print("--------------------------------")
    print(verification_result_dict)

    return render_template("result2.html", **verification_result_dict)

if __name__ == "__main__":
    app.run(debug=True)
