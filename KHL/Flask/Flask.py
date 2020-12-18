from flask import Flask
import blueprint

def create_app():
    app = Flask(__name__)

    app.register_blueprint(blueprint.bp)
    
    return app
