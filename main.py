from flask import Flask, redirect, url_for, request, render_template, jsonify

app=Flask(__name__)

@app.route("/<name>")
def index(name):
    return render_template("layout.html", content=name)

@app.route('/about')
def about():
    return '<h1>About Us!</h1>'

@app.route('/contact')
def contact():
    return '<h1>Contact Us!</h1>'

@app.route('/admin')
def admin():
    return redirect(url_for("index"))

@app.route("/<name>")
def user(name):
    return f"Hello {name}!"

@app.route("/test")
def test():
    return redirect(url_for("user", name="test!"))

@app.route("/<not_found>")
def notfound(not_found):
    return render_template("notfound.html", page=not_found)
# app.run(debug=True)
