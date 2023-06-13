from flask import Flask, request, jsonify
from threading import Thread
from infer_final import infer_final
import os

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def serve():
    args = request.args
    code = '200'
    message = 'success'
    if 'uuid' in args:
        uuid = request.args.get('uuid')
    else:
        code = '400'
        message = 'Illegal Arguments: must contain attribute \'uuid\''
    if 'question' in args:
        question = request.args.get('question')
    else:
        code = '400'
        message = 'Illegal Arguments: must contain attribute \'question\''
    if code == '400':
        return jsonify(uuid=uuid, code=code, message=message, data='')
    print(question)
    answer = infer_final(question)
    print(answer)
    return jsonify(uuid=uuid, code=code, message='success', data=answer)


def run():
    app.run(host='0.0.0.0', port=5000)


if __name__ == '__main__':
    thread = Thread(target=run)
    thread.start()
    os.system("ngrok http 5000")



"""
serveo.net
ssh -R 80:localhost:3000 serveo.net
"""