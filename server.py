from flask import Flask, request
import json
import os
from pathlib import Path
import sys

#1. flask instance
app = Flask(__name__)

#2 model prediction function
#content and style are the paths for the content and style image respectively

def return_prediction(model,content,style):
    styledImgB64 = model.style_image(content, style)
    return {"image": styledImgB64 }
def apply_effect(applier,filter,image):
    res= applier.return_result(filter,image)
    return {"image": res}

#3 load the model
import style_transfer
import opencv_effects

#4Setting up the page of response
@app.route("/style_transfer_reponse", methods=['GET','POST'])
def style_response():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        imgs = request.json
        content = imgs['input']
        style = imgs['style']
        results= return_prediction(style_transfer,content,style)
        return results
    else:
        content='/home/kyoraku/Pictures/Pano_Robley2.jpg'
        style='/home/kyoraku/Pictures/geometric-composition-artworks-von-danielle.jpg'
        results= return_prediction(style_transfer,content,style)
        return results

#4Setting up the page of response of the filters
@app.route("/filters", methods=['GET','POST'])
def filter_response():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        req = request.json
        image = req['input']
        filter = req['filter']
        results= apply_effect(opencv_effects,filter,image)
        return results
    else:
        image='/home/kyoraku/Pictures/Pano_Robley2.jpg'
        filter='oil'
        results= apply_effect(opencv_effects,filter,image)
        return results



#4Setting up the page of response
@app.route("/")
def home():
    return """
    <h1>Welcome to our humble server<h1>
    To see the result of our model make a post request in the url /style_transfer_reponse with the following fields :
    <li>content_image:'path of the image'
    <li>style_image:'path of the image'
    <p>The response for the style trasfer is a json with the format :<p>
    <li> {image:'based64 text'} (this needed to be converted to an image to be shown) 
    """
if __name__ == '__main__':
    app.run(debug=True)