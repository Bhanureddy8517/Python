import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from flask import *

app=Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/viewdata',methods=["GET","POST"])
def viewdata():
    dataset = pd.read_csv(r'stress_detection_IT_professionals_dataset.csv')
    dataset.to_html()
    print(dataset)
    print(dataset.head(2))
    print(dataset.columns)
    return render_template("viewdata.html", columns=dataset.columns.values, rows=dataset.values.tolist())

@app.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global x, y, x_train, x_test, y_train, y_test,  df
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100
        df=pd.read_csv(r'stress_detection_IT_professionals_dataset.csv')

        ##splitting
        x=df.drop('Stress_Level',axis=1)
        y=df['Stress_Level']

        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=size,random_state=42)

        # describes info about train and test set
        print("Number transactions X_train dataset: ", x_train.shape)
        print("Number transactions y_train dataset: ", y_train.shape)
        print("Number transactions X_test dataset: ", x_test.shape)
        print("Number transactions y_test dataset: ", y_test.shape)

    
        print(x_train,x_test)

        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')
    return render_template('preprocess.html')

@app.route('/model', methods=["POST","GET"])
def model():
    if request.method=="POST":
        global model
        s=int(request.form['algo'])
        if s==0:
            return render_template('model.html',msg="Choose an algorithm")
        elif s==1:
            
            rf=RandomForestRegressor()
            rf.fit(x_train,y_train)
            y_pred=rf.predict(x_test)
            ac_rf=r2_score(y_pred,y_test)
            ac_rf=ac_rf*100
            msg="The r2_score obtained by RandomForestRegressor is "+str(ac_rf) + str('%')
            return render_template("model.html",msg=msg)
        elif s==2:
            
            ad = AdaBoostRegressor()
            ad.fit(x_train,y_train)
            y_pred=ad.predict(x_test)
            ac_ad=r2_score(y_pred,y_test)
            ac_ad=ac_ad*100
            msg="The r2_score obtained by AdaBoostRegressor "+str(ac_ad) +str('%')
            return render_template("model.html",msg=msg)
        elif s==3:
            
            ex = ExtraTreeRegressor()
            ex.fit(x_train,y_train)
            y_pred=ex.predict(x_test)
            ac_dt=r2_score(y_pred,y_test)
            ac_dt=ac_dt*100
            msg="The r2_score obtained by ExtraTreeRegressor is "+str(ac_dt) +str('%')
            return render_template("model.html",msg=msg)
        
    return render_template("model.html")


@app.route('/prediction' , methods=["POST","GET"])
def prediction():
    if request.method=="POST":
        f1=float(request.form['Heart_Rate'])
        f2=float(request.form['Skin_Conductivity'])
        f3=float(request.form['Hours_Worked'])
        f4=float(request.form['Emails_Sent'])
        f5=float(request.form['Meetings_Attended'])
        

        lee=[f1,f2,f3,f4,f5]
        print(lee)

        
        model=RandomForestRegressor()
        model.fit(x_train,y_train)
        result=model.predict([lee])
        print(result)
        msg="The Stress level IT Professionals  is "+str(result) +str('%')
        return render_template('prediction.html', msg=msg)
    return render_template("prediction.html") 



if __name__=="__main__":
    app.run(debug=True)