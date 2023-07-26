from app import app
from flask import request, render_template
from datetime import datetime
from utils.main import main


@app.route('/',methods=['GET'])
def index():
   return render_template('index.html')

@app.route('/view/<video_filename>',methods=['GET'])
def preview(video_filename):
   return render_template('preview.html',data=video_filename)

@app.route('/up',methods=['POST'])
def upload():
   now=datetime.now()
   now_unix=int(datetime.timestamp(now))
   file=request.files['file']
   filename=f"{now_unix}-{file.filename}"
   file.save(filename)
   print("selesai di upload")
   jalan=main(filename)
   
   
   return f"<a href='{jalan}' download>Download Hasil</a>", 200


