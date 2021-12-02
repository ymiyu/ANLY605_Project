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
    path = 'static/Boxplot.jpeg' #Have to edit this line!
    request_type_str = request.method
    if request_type_str == "GET":
        return render_template('index.html', href=path)
    else:
        text = request.form['text']
        random_string = uuid.uuid4().hex
        path = "static/" + random_string + ".svg"
        #return render_template('index.html', href='static/Boxplot.jpeg')
        
        # read user input
        np_arr = floatsome_to_np_array(text).reshape(1,-1)
        pkl_filename="TrainedModel/stacking_model.pkl"
        with open(pkl_filename, 'rb') as file:
            pickle_model = pickle.load(file)
        plot_graphs(model=pickle_model, new_input_arr=np_arr, output_file = path)
        return render_template("index.html",href=path)

def floatsome_to_np_array(float_str):
    def is_float(s):
        try:
            float(s)
            return True
        except:
            return False
    floats = np.array(
        [
            float(x) for x in float_str.split(',') if is_float(x)
        ]
    )
    return floats.reshape(len(floats),1)



def plot_graphs(model,new_input_arr, output_file):
    df = pd.read_csv("bank-additional-full.csv")

    fig = make_subplots(
    rows=1, cols=2
    # subplot_titles=("Plot 1", "Plot 2", "Plot 3", "Plot 4")
    )

    fig.add_trace(
        go.Scatter(x=df["age"],y=df['y'],mode='markers',
        marker=dict(
                color="#003366"),
            line=dict(color="#003366",width=1)),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=df['duration'],y=df['y'],mode='markers',
        marker=dict(
                color="#FF6600"),
            line=dict(color="#FF6600",width=1)),
        row=1, col=2
    )

    new_preds = model.predict(new_input_arr)
    # print(new_preds)
    RM_input = np.array(new_input_arr[0][5])
    # print(RM_input)
    LSTAT_input =np.array(new_input_arr[0][12])
    # print(LSTAT_input)

    fig.add_trace(
    go.Scatter(
        x=LSTAT_input,
        y=new_preds,
        mode='markers', name="Predicted Output",
        marker=dict(
            color="#FFCC00",size=15),
        line=dict(color="#FFCC00",width=1)),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=RM_input,
            y=new_preds,
            mode='markers', name="Predicted Output",
            marker=dict(
                color="#6600cc",size=15),
            line=dict(color="red",width=1)),
            row=1, col=2
    )

    # Update xaxis properties
    fig.update_xaxes(title_text="Age", row=1, col=1)
    fig.update_xaxes(title_text="Call Duration", row=1, col=2)

    # Update yaxis properties
    fig.update_yaxes(title_text="Yes/No", row=1, col=1)
    # fig.update_yaxes(title_text="yaxis 2 title", range=[40, 80], row=1, col=2)
    # Update title and height
    fig.update_layout(height=600, width=1400, title_text="Client campaing acceptance")
    output_file="app/static/scatterplot.svg"
    fig.write_image(output_file,width=1200,engine="kaleido")
    fig.show()

