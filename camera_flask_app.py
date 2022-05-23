from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
from pytesseract import Output
import pytesseract
import argparse
import cv2
import numpy as np
import re
#from tensorflow.keras.models import load_model
pytesseract.pytesseract.tesseract_cmd = r'/Users/bhavyaprafulthakker/opt/anaconda3/bin/tesseract'
protein_dict = ['protein','pratein','prtein','proein','praein'] 
carbs_dict = ['carbohydrates','carbohydnates','carbohydntes','corbahydrates','carb','arbohydrates']
fat_dict = ['Fat','Ft','Fot','Fdt']
sugar_dict = ['sugar','setee','suar','su','sugas']
fiber_dict = ['fiber','fibor','fiben']
sodium_dict=['sodium','sotium','sotem','sodiem','sotiem']
potassium_dict = ['potassium','porassium']
calories_dict=['galories','calories','kcal','cal']

global capture,rec_frame, grey, switch, neg, face, rec, out 
capture=0
grey=0
neg=0
face=0
switch=1
rec=0

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

#Load pretrained face detection model    
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


camera = cv2.VideoCapture(0)

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)


def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))   
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:            
            return frame           

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame=frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = ( int(w * r), 480)
        frame=cv2.resize(frame,dim)
    except Exception as e:
        pass
    return frame
 

def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        if success:
            if(face):                
                frame= detect_face(frame)
            if(grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if(neg):
                frame=cv2.bitwise_not(frame)    
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
                mainOCRFunction(p)
                #call your function here 
            
            if(rec):
                rec_frame=frame
                frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                frame=cv2.flip(frame,1)
            
                
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass
def mainOCRFunction(path):
    img = cv2.imread(path,0)
    #Contrasting and thresholding the image
    #increasing contrast by spreading the intensities a.k.a spreading the histogram of intensities
    #dst = cv2.equalizeHist(img.copy())
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    #reducing noise while maintaining the sharpness around edges
    dst = cv2.bilateralFilter(cl1.copy(),5,70,70)
    #applying local threshholding to get clearer tesxt back
    block_size=int(0.1*img.shape[0])
    if block_size%2==0:
        block_size=block_size-1
    th = cv2.adaptiveThreshold(dst.copy(),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,block_size,15)
    #finding all contours
    cnts,heirachy = cv2.findContours(th.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    img2 = img.copy()
    #plotting back all contours
    cnt_img = cv2.drawContours(img2,cnts,-1,(255,0,0),3)

    #filtering closed contours and selecting the one with highest perimeter assuming nutrition label would be the focus in image
    #making a list of perimeters of closed contours
    perimeter=[ cv2.arcLength(c,True) for c in cnts]
    #justplain white background
    mask = np.ones(shape=img.shape[:2])*255
    #assigning nutrition label contour
    possiblelabels = cnts[np.argmax(perimeter)]
    cv2.drawContours(mask,[possiblelabels],-1,(0,266,0),3)
    #plt.imshow(mask)
    cv2.imwrite('testaoi.jpg',mask)

    #using affine transformation to zoom in on the area of interest
    #fitting a rectangle and to nutrition label contour
    x,y,w,h = cv2.boundingRect(possiblelabels)
    #plt.imshow(mask)
    cv2.imwrite('testaoi.jpg',mask)

    #using affine transformation to zoom in on the area of interest
    #fitting a rectangle and to nutrition label contour
    x,y,w,h = cv2.boundingRect(possiblelabels)
    #creating points for transformation x & y initial are initial x & y co-ordinates respectively then for transformation we assume x & y
    #values as 0 then we get transformation points
    pts1 = np.float32([[x,y],[x,y+h],[x+w,y],[x+w,y+h]])
    pts2 = np.float32([[0,0],[0,0+h],[0+w,0],[0+w,0+h]])
    #creating transformation matrix
    m = cv2.getPerspectiveTransform(pts1,pts2)
    #applying it image with w,h resolution
    final = cv2.warpPerspective(th,m,(w,h))
    final2 = cv2.bilateralFilter(final.copy(),5,70,70)


    cv2.imwrite('data/final.jpg',final2)

    cv2.imwrite('data/testgs.jpg',img)

    cv2.imwrite('data/testwth.jpg',dst)

    cv2.imwrite('data/testth.jpg',th)

    cv2.imwrite('data/testcnt.jpg',cnt_img)

    stro = pytesseract.image_to_string(img)
    fo = open('data/datao.txt','w')
    fo.write(stro)
    fo.close()

    strcnt = pytesseract.image_to_string(dst)
    fcnt = open('data/datacnt.txt','w')
    fcnt.write(strcnt)
    fcnt.close()

    str = pytesseract.image_to_string(final2)
    f = open('data/data.txt','w')
    f.write(str)
    f.close()
    x_exmp = best_params(org='data/datao.txt',contrast='data/datacnt.txt',final='data/data.txt')
# function that takes all three file names and finds the best parmeters
def best_params(org,contrast,final):
    f_count = 0
    c_count = 0
    o_count = 0
    og = {}
    cn = {}
    f = {}
    f_count, f = normalize_vals(final)
    c_count, cn = normalize_vals(contrast)
    o_count, og =normalize_vals(org)
    if ((f_count>=c_count) and (f_count>=o_count)):
        return f
    elif(c_count>=o_count):
        return cn
    else:
        return og
def normalize_vals(filename):
    values = get_info(filename)
    cor_count = 7
    for i in values:
        if(i == "fat" and values[i] == 1001):
            values[i] = 10
            cor_count = cor_count -1
        if(i == "sodium" and values[i] == 1001):
            values[i] = 100
            cor_count = cor_count -1
        if(values[i]== "fiber" and values[i] == 1001):
            values[i] = 2
            cor_count = cor_count -1
        if(i == "carbs" and values[i] == 1001):
            values[i] = 40
            cor_count = cor_count -1
        if(i == "sugar" and values[i] == 1001):
            values[i] = 10
            cor_count = cor_count -1
        if(i == "potassium" and values[i] == 1001):
            values[i] = 10
            cor_count = cor_count -1
        if(i == "protein" and values[i] == 1001):
            n_p_c=(values[1]*2+10*4)/((values[2]*4)+(10*4)+(values[0]*9))
            values.insert(7,n_p_c)
            values[i] = (10/100)
            cor_count = cor_count -1
        if((i==1 or i==5) and values[i] != 1001):
            values[i] = values[i]/1000
        if(i==6 and values[i] != 1001):
            n_p_c = ((values[1]*2) + (values[6]*4))/((values[2]*4) + (values[6]*4) + (values[0]*9))
            values.insert(7,n_p_c)
            values[i] = values[i]/100
        if((i !=6 and (i!=1 and i!=5)) and values[i]!=1001):
            values[i] = values[i]/100
    return cor_count,values


def get_info(filename):
    vals={}
    f = open(filename,'r')
    for x in f:
        if x.isspace():
            pass
        else:
            vals = add_vals(fat_dict,0,x,vals,"fat")
            vals = add_vals(sodium_dict,1,x.casefold(),vals,"sodium")
            vals = add_vals(fiber_dict,2,x.casefold(),vals,"fiber")
            vals = add_vals(carbs_dict,3,x.casefold(),vals,"carbs")
            vals = add_vals(sugar_dict,4,x.casefold(),vals,"sugar")
            vals = add_vals(potassium_dict,5,x.casefold(),vals,"potassium")
            vals = add_vals(protein_dict,6,x.casefold(),vals,"protein")
    return vals
def add_vals(patterns,pos,txt,vals,key):
    for i in patterns:
        if (txt.find(i)) != -1:
            txt = txt[txt.find(i):]
            val = re.search("[0-9]{2}|[0-9]{1}|[0-9]{3}",txt)
            if val:
                vals[key] = int(val.group())
                #vals[pos] = int(val.group())
                break
            else:
                vals[key] = 1001
        else:
            pass
    return vals

@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('grey') == 'Grey':
            global grey
            grey=not grey
        elif  request.form.get('neg') == 'Negative':
            global neg
            neg=not neg
        elif  request.form.get('face') == 'Face Only':
            global face
            face=not face 
            if(face):
                time.sleep(4)   
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
        elif  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec= not rec
            if(rec):
                now=datetime.datetime.now() 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                #Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()
            elif(rec==False):
                out.release()
                          
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows()     