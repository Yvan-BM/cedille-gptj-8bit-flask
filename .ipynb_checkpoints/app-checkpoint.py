from flask import Flask, jsonify, request
import io
import json
import function as usr_src
import os

usr_src.init()

app = Flask(__name__)

# Reference your preloaded global model variable here.
@app.route('/generate', methods=['POST'])
def generate():
    
    if request.method == 'POST':

        # Parse out your arguments
        prompt = request.form.get('prompt')
        
        if prompt == None:
            return {'message': "No prompt provided"}
        
        result = usr_src.inference(prompt.strip())

        # Return the results as a dictionary
        return result



if __name__ == '__main__':
    app.run(host='0.0.0.0', port="8000")