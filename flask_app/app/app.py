from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
import plotly.express as px
import plotly.subplots as make_subplots
import plotly.graph_objs as go
import uuid

app = Flask(__name__)

@app.route("/", methods=['GET','POST'])
def hello_world():
    #return render_template('index.html', href='static/Boxplot.jpeg'
    #path = 'static/barplot1.svg' #Have to edit this line!
    request_type_str = request.method
    if request_type_str == "GET":
        return render_template('index.html', href="static/barplot1.svg")
    else:
        #return render_template('index.html', href='static/Boxplt.jpeg')
        text = request.form['text']
        random_string = uuid.uuid4().hex
        path = "static/" + random_string + ".svg"
        #return render_template('index.html', href='static/Boxplot.jpeg')
        
        # read user input
        np_arr = get_input(text)
        pkl_filename="TrainedModel/pipeline.pkl"
        with open(pkl_filename, 'rb') as file:
            pickle_model = pickle.load(file)
        plot_graphs(model=pickle_model, new_input_arr=np_arr, output_file = path)
        return render_template("index.html",href=path)


def get_input(string):
    lis = [str(x).strip() for x in string.split(',')]
    df = pd.DataFrame(lis).transpose()
    df.columns = ["age","job","marital","education","default",
                    "housing","loan","contact","duration",
                    "campaign","pdays","previous","poutcome",
                    "emp.var.rate","cons.price.idx","cons.conf.idx",
                    "euribor3m","nr.employed"]
    df['age'] = df.age.astype('int')
    df["job"] = df.job.astype("category")
    df["marital"] = df.marital.astype("category")
    df["education"] = df.education.astype("category")
    df["default"] = df.default.astype("category")
    df["housing"] = df.housing.astype("category")
    df["loan"] = df.loan.astype("category")
    df["contact"] = df.contact.astype("category")
    df['duration'] = df.duration.astype('int')
    df['campaign'] = df.campaign.astype('int')
    df['pdays'] = df.pdays.astype('int')
    df['previous'] = df.previous.astype('int')
    df["poutcome"] = df.poutcome.astype("category")
    df['emp.var.rate'] = df['emp.var.rate'].astype('float')
    df['cons.price.idx'] = df['cons.price.idx'].astype('float')
    df['cons.conf.idx'] = df['cons.conf.idx'].astype('float')
    df['euribor3m'] = df['euribor3m'].astype('float')
    df['nr.employed'] = df['nr.employed'].astype('float')
    

    return df




def plot_graphs(model,new_input_arr, output_file):
    df = pd.read_csv("bank-additional-full.csv", sep=";")
    df_age = df[["age","y"]]
    df_age["age_group"] = (df_age["age"]//10)*10
    df_age = df_age.groupby(["age_group","y"], as_index=False).count()


    new_preds = model.predict(new_input_arr)[0]

    # Get the age from user input
    age = int(new_input_arr["age"])
    age_group = age//10 -1

    # Calculate the correct ba to highlight
    highlight = [0,0,0,0,0,0,0,0,0,0]
    age_group = age//10 -1
    highlight[age_group] = 4
    

    trace1 = go.Bar(
        x=df_age[df_age["y"]=="yes"]["age_group"],
        y=df_age[df_age["y"]=="yes"]["age"],
        name='Accepted'
    )
    trace2 = go.Bar(
        x=df_age[df_age["y"]=="no"]["age_group"],
        y=df_age[df_age["y"]=="no"]["age"],
        name='Declined'
    )

    data1 = [trace1, trace2]
    layout = go.Layout(
        barmode='stack'
    )

    # Draw border around last bars
    for bar in [data1[new_preds]]:
        bar.marker.line.color = 'green'
        bar.marker.line.width = highlight

    fig = go.Figure(data=data1, layout=layout)

    # Update xaxis properties
    fig.update_xaxes(title_text="Age")

    # Update yaxis properties
    fig.update_yaxes(title_text="Count")
    # fig.update_yaxes(title_text="yaxis 2 title", range=[40, 80], row=1, col=2)
    # Update title and height
    fig.update_layout(height=600, width=1400, title_text="Client campaing acceptance")
    fig.write_image(output_file,width=1200,engine="kaleido")
    fig.show()
