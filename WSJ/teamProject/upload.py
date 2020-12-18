from flask import Flask, render_template, request
app = Flask(__name__)
import CV07_Use_multiple as cv07

#업로드 HTML 렌더링
@app.route('/')
def render_file():
   return render_template("upload.html")

# #파일 업로드 처리
@app.route('/file', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      time = request.form['time']
      fileName = f.filename
      # f.save("./teamProject/video/"+fileName)
      # f.save("./teamProject/static/"+fileName)
      # cv07.dd(fileName,time)
      # ff = "dd"
      return render_template("show.html")

if __name__ == '__main__':
    #서버 실행
   app.run()
