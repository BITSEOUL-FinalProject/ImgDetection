from flask import Flask
from flask import Blueprint

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/hello')                 # 주소 뒤에 붙으면 해당 이름 함수내용이 실행된다
def home():
    return 'Hello, World!'

@bp.route('/')
def index():
    return "index"
