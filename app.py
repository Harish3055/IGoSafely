"""
Created on Fri Dec 11 12:46:56 2020

@author: Harish_3055
Team Name    :- Mission I'mpossible
Project Name :- bSafe
Project Desc :- In the recent past we witnessed a lot of vulnerability against women on various occasions.
                To safeguard women we have come up with an idea to build a better environment. Taking 
                into account the google news source of data about crime we have classified the news based 
                on crime rate. So people can use our app to know more about a location and safeguard them
                from danger.
Packages used:-
Flask       ->  We have used the flask to connect the html part and the python part.
flask_ngrok ->  To make the Flask apps running on localhost available over the internet via the excellent ngrok tool.
tensorflow  ->  To load the trained model and to get the predicted result.
GoogleNews  ->  To get the news result on a particular date, location, Country.   
"""
from cgitb import text
import pickle
import requests
from bs4 import BeautifulSoup as soup
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
#Loaded the trained NLP classification model to get the crime rate of the news
model = tf.keras.models.load_model('./model_news.h5')

with open('./tokenizer.pickle', 'rb') as handle:
    vec= pickle.load(handle)


def getNewsData(data):
  data = data.replace(' ','%20')
  link = "https://news.google.com/search?q={}&hl=en-IN&gl=IN&ceid=IN%3Aen".format(data)
  resp=requests.request(method="GET",url=link)
  soup_parser = soup(resp.text, "html.parser")
  ls = soup_parser.find_all("a", class_="DY5T1d")[:50]
  data=[]
  for i in ls:
        data.append([i.text[:500],'https://news.google.com'+i.get('href')[1:]])
  return data

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')#To display the details in html

@app.route('/news',methods=['POST'])
def news():
    country=request.form['country']#to get country
    location=request.form['location']#to get location
    start_date=request.form['start_date']#to get start 
    end_date=request.form['end_date']#to get end date
    
    #to covert yyyy-mm-dd to mm-dd-yyyy
    start_date = start_date[3:5]+"-"+start_date[5:]+"-"+start_date[:3]
    end_date=end_date[3:5]+"-"+end_date[5:]+"-"+end_date[:3]
    start_date=start_date[3:]+start_date[0]
    end_date=end_date[3:]+end_date[0]
    head={}
    link={}
    color=[]
    for data in getNewsData(location+" crime against woman"):#To predict the correct crime rate for the particular news result 
        if data[0] != ['']:
            key = pad_sequences(vec.texts_to_sequences(data[0]), maxlen=10, padding='post', truncating='post')
            res = model.predict(key)[0][0]*100
            head[res] = data[0]
            link[res] = data[1]
    print()
    
    head = dict(sorted(head.items(), key=lambda item: item[0],reverse=True))
    link = dict(sorted(link.items(), key=lambda item: item[0],reverse=True))
    for i in list(head.keys()):
        if i>=75:
            color.append('red')
        elif i>=50 and i<75:
            color.append('orange')
        else:
            color.append('Green')
            
    #To display the predicted output in output.html
    return render_template('output.html',a=list(head.values()),leng=len(head),pred=list(head.keys()),col=color,link=list(link.values()))
if __name__== '__main__':
    app.run(debug=True)