#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from math import sqrt 
from sklearn.metrics import r2_score
import dash
import datetime
import pandas as pd
import plotly.express as px
from dash import html ,dcc,Dash
from jupyter_dash import JupyterDash
import matplotlib.pyplot as plt
from dash.dependencies import Input, Output,State
import dash_bootstrap_components as dbc 
from plotly.subplots import make_subplots
import plotly.graph_objects as go 
import dash_daq as daq
import plotly.figure_factory as ff
from Adam import Adam
from BFGS import BFGS

# In[2]:


data = pd.read_csv('Weather.csv')


# In[3]:


X = data.iloc[:, 0]
y = data.iloc[:, -1]
y = y.values.reshape(-1,1)


# In[4]:


X = X.values.reshape(-1, 1)


# In[5]:


# In[6]:


# In[7]:


app = JupyterDash(external_stylesheets=[dbc.themes.BOOTSTRAP,dbc.icons.BOOTSTRAP])
server = app.server
app.layout = html.Div([
    
    html.Div(
          
    # Header
    html.H4('Numerical Optmization',style={'color':'rgba(255, 255, 255, 0.7)',
                                           'background':'#183D4E',
                                           'font-size':'100px',
                                           'font-family':'Helvetica',
                                           'line-height': '1.45em',
                                           'font-weight': '300',
                                           'margin-left': 'auto',
                                           'margin-right': 'auto',                                          
                                           'text-align': 'center'})),
    
                       
    dcc.Tabs(id = "tabs-example-graph", value = 'tab-1-example-graph', children = 
      [       
        dcc.Tab(children=[
            
            dbc.Row([html.Div(html.Br())]),

            dbc.Row([  
                  
                    # Button           
                    dbc.Col([daq.PowerButton(id='our-power-button-1', on = False, size = 100, color = 'rgba(24, 61, 78)')]),
                    
            # Optimizer Parameters
                    
                    # Learning rate                
                    dbc.Col([html.H4("Learning rate" , style = {'color' : '#48cae4', 'font-size' : '25px'}),
                            dcc.Dropdown([0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10], value= 0.01,id = "lr", style = {'width' : '200px'}),]),
                    # Iterations    
                    dbc.Col([html.H4("Iteration" , style={'color' : '#48cae4','font-size' : '25px'}),
                            dcc.Input(id="iteration", type="number", placeholder="",value= 1000, style={'marginRight':'10px'}),]), 
                    
                    dbc.Col([html.H4("Beta1" , style={'color':'#48cae4','font-size':'25px'}),
                            dcc.Dropdown([0.07,0.08,0.99,0.09,0.9,0.1,0.5,1],id = "Beta1",value= 0.9, style={'width':'200px'}),]),
        
                    dbc.Col([html.H4("Beta2" , style={'color':'#48cae4','font-size':'25px'}),
                            dcc.Dropdown([0.07,0.08,0.99,0.09,0.9,0.1,0.5,1],id = "Beta2",value= 0.9, style={'width':'200px'})])
                            
                  ],style={'margin-left': '50px', 'margin-right': '50px','background-color': '#f7f7f7'}), 
# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ
         
             # Score
            dbc.Row([html.H2(id='score',style={'color':'#006d77','font-size':'52px', 'text-align': 'center'})],
                    style={'margin-left': '50px', 'margin-right': '50px','background-color': '#f7f7f7'}),
            # Number of epochs
            dbc.Row([html.H2(id='epochs',style={'color':'#006d77','font-size':'56px', 'text-align': 'center'})], 
                    style={'margin-left': '50px', 'margin-right': '50px','background-color': '#f7f7f7'}),  
                

            dbc.Row([html.Div(html.Br())],style={'margin-left': '50px', 'margin-right': '50px','background-color': '#f7f7f7'}),
    
            
# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ
        # Graphs:
            
            dbc.Row([
                #### loss
                dbc.Col([dcc.Graph(id = 'Loss'), dcc.Interval(id='Loss interval',interval= 400, max_intervals= 5000, disabled= True)]),
                ### linear regression
                dbc.Col([dcc.Graph(id = 'Regression_line'),dcc.Interval(id='Loss interval',interval= 400, max_intervals= 5000, disabled= True)]), 
                #### theta     
                dbc.Col([dcc.Graph(id = 'Theta Graph'),dcc.Interval(id='Loss interval',interval= 400, max_intervals= 5000, disabled= True)]),   
                ], style={'margin-left': '50px', 'margin-right': '50px','background-color': '#f7f7f7'}),
    
# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ
         # Columns visualization:
            
            dbc.Row([
                
                dbc.Col([
                        # Features 
                        html.H4("Coulmns Name", style = {'color' : '#48cae4', 'font-size' : '25px'}),
                        dcc.Dropdown(['MaxTemp', 'MinTemp'], id = 'demo-dropdown1', style = {'width':'300px'}),
                
                        # Chart Type
                        html.H4("Chart Type" , style = {'color' : '#48cae4', 'font-size' : '25px'}),
                                dcc.Dropdown(['Scatter', 'Distplot'], id = 'demo-dropdown2', style = {'width' : '300px'}),

                        # Visualize Button 
                        dbc.Row([html.Div(
                                html.Br())]),
                                html.Button('Visualize', id = 'button-example-2',
                                style= {'color': '#48cae4', 'font-size' : '42px', 'text-align' : 'center',
                                        'background': '#457b9d', 'width': '300px', 'text-align': 'center' }),      
                        ], style = {'padding' : '50px'}),

                # Visualization graph
                dbc.Col([dcc.Graph(id = 'visualization feature')]),     
            ],style={'margin':'50PX','background-color': '#f7f7f7'})
  

      ], label = 'ADAM WITH MULTIVARUABLE', value = 'tab-1-example-graph',
        style={'color': '#f1faee', 'background': '#a2d2ff', 'font-size': '40px'}), 

# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ
       # BFGS: 
# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ

        dcc.Tab(children=[
            
            dbc.Row([html.Div(html.Br())]),

            dbc.Row([  
                  
                    # Button           
                    dbc.Col([daq.PowerButton(id='BFGS_our-power-button-1', on = False, size = 100, color = 'rgba(24, 61, 78)')]),
                    
            # Optimizer Parameters
                    
                    # Iterations    
                    dbc.Col([html.H4("Iteration" , style={'color' : '#48cae4','font-size' : '25px'}),
                            dcc.Input(id="BFGS_iteration", type="number", placeholder="", value= 10,style={'marginRight':'10px'}),]),
                            
                  ],style={'margin-left': '50px', 'margin-right': '50px','background-color': '#f7f7f7'}), 
# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ
         
             # Score
            dbc.Row([html.H2(id='BFGS_score',style={'color':'#006d77','font-size':'52px', 'text-align': 'center'})],
                    style={'margin-left': '50px', 'margin-right': '50px','background-color': '#f7f7f7'}),
            
            # Number of epochs
            dbc.Row([html.H2(id='BFGS_epochs',style={'color':'#006d77','font-size':'56px', 'text-align': 'center'})], 
                    style={'margin-left': '50px', 'margin-right': '50px','background-color': '#f7f7f7'}),  
                

            dbc.Row([html.Div(html.Br())],style={'margin-left': '50px', 'margin-right': '50px','background-color': '#f7f7f7'}),
    
            
# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ
        # Graphs:
            
            dbc.Row([
                #### loss
                dbc.Col([dcc.Graph(id = 'BFGS_Loss'), dcc.Interval(id='BFGS_Loss interval',interval= 400, max_intervals= 5000, disabled= True)]),
                ### linear regression
                dbc.Col([dcc.Graph(id = 'BFGS_Regression_line'),dcc.Interval(id='BFGS_Loss interval',interval= 400, max_intervals= 5000, disabled= True)]), 
                #### theta     
                dbc.Col([dcc.Graph(id = 'BFGS_Theta Graph'),dcc.Interval(id='BFGS_Loss interval',interval= 400, max_intervals= 5000, disabled= True)]),   
                ], style={'margin-left': '50px', 'margin-right': '50px','background-color': '#f7f7f7'}),
            

# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ
         # Columns visualization:
            
            dbc.Row([
                
                dbc.Col([
                        # Features 
                        html.H4("Coulmns Name", style = {'color' : '#48cae4', 'font-size' : '25px'}),
                        dcc.Dropdown(['MaxTemp', 'MinTemp'], id = 'BFGS_demo-dropdown1', style = {'width':'300px'}),
                
                        # Chart Type
                        html.H4("Chart Type" , style = {'color' : '#48cae4', 'font-size' : '25px'}),
                                dcc.Dropdown(['Scatter', 'Distplot'], id = 'BFGS_demo-dropdown2', style = {'width' : '300px'}),

                        # Visualize Button 
                        dbc.Row([html.Div(
                                html.Br())]),
                                html.Button('Visualize', id = 'BFGS_button-example-2',
                                style= {'color': '#48cae4', 'font-size' : '42px', 'text-align' : 'center',
                                        'background': '#457b9d', 'width': '300px', 'text-align': 'center' }),      
                        ], style = {'padding' : '50px'}),

                # Visualization graph
                dbc.Col([dcc.Graph(id = 'BFGS_visualization feature')]),     
            ],style={'margin': '50px', 'background-color': '#f7f7f7'})
  

      ],label='BFGS WITH MULTIVARUABLE', value='tab-2-example-graph',
                    style={'color':'#f1faee','background':'#a2d2ff','font-size':'40px'}),    
    ]),

    # Adam or BFGS
    html.Div(id = 'tabs-content-example-graph')
])


# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ
#                                                               Changing Tabs 
# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ

@app.callback(Output('tabs-content-example-graph', 'children'),
              Input('tabs-example-graph', 'value'))
def render_content(tab):
    if tab == 'tab-1-example-graph':       
        return html.Div([])
    
    elif tab == 'tab-2-example-graph':
        return html.Div([])


# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ
#  ADAM CallBacks
# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ

# to live counting epochs 
@app.callback(
    Output('Loss interval', 'disabled'),
    Input('our-power-button-1', 'on'),
    State('Loss interval', 'disabled')
)
def toggle_interval(n_clicks, interval_disabled):
    if n_clicks:
        return False
    else:
        return True
# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ

# Reset the interval index to 0 after power button = off 
@app.callback(
        Output('Loss interval', 'n_intervals'),
        Input('our-power-button-1', 'on'),
              )
def update_dashboard(Button):
    if ~Button :
        return 0 
# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ

@app.callback(
    Output('Loss', 'figure'),
    Output(component_id='Regression_line', component_property = 'figure'),
    Output(component_id='Theta Graph', component_property = 'figure'),
    Output(component_id='score', component_property = 'children'),
    Output(component_id='epochs', component_property = 'children'), 
    Output('Loss interval', 'max_intervals'),
    Input('Loss interval', 'n_intervals'),
    Input(component_id = 'our-power-button-1', component_property = 'on'),
    State(component_id = 'lr', component_property = 'value'),
    State(component_id = 'iteration', component_property = 'value'),
    State(component_id = 'Beta1', component_property = 'value'), 
    State(component_id = 'Beta2', component_property = 'value'),
)
def update_graph(n1,Button, alpha, iteration, beta1, beta2):
  if Button:
      batch_size = len(X)
      alltheta, total_pred, loss, J, y_pred, i, theta = Adam(X, y, iteration, alpha, beta1, beta2, batch_size)
      fig1 = px.line(loss[:n1])

      fig1.update_layout({
                      'title': {'text':'Loss Vs Number of Iterations','x': 0.5,'y': 0.9},
                      'xaxis': {'title': {'text': 'Iterations'}},
                      'yaxis': {'title': {'text': 'Loss'}},
                      }) 
      fig2 = go.Figure()
      fig2.add_trace(go.Scatter(x = X.reshape(-1), y = y.reshape(-1), mode='markers', name = 'Data'))
      fig2.add_trace(go.Scatter(x = X.reshape(-1), y = total_pred[n1].reshape(-1,1).reshape(-1), name = 'Regression Line'))
      fig2.update_layout({'title': {'text':'Regression Line','x': 0.5,'y': 0.9}})

      fig3 = make_subplots(rows=2, cols=1,
                          subplot_titles=['theta0','theta1'])

      fig3.add_trace(go.Scatter(x = alltheta[:n1][:, 0], y = loss, name = 'theta0'), row = 1, col = 1)
      fig3.add_trace(go.Scatter(x = alltheta[:n1][:, 1], y = loss, name = 'theta1'), row = 2, col = 1)

      fig3.update_layout({'title': {'text':'Plots of Thetas Vs Loss','x': 0.5,'y': 0.9}})
      score = np.round((r2_score(y, total_pred[n1]) * 100), 2)
       
      return fig1,fig2, fig3, f'the score is {score}%',f'Gradient Descent converged after {n1+1} epochs',i 
    
  else:     
      fig1 = px.scatter(x = np.arange(len(X)), y = np.arange(len(X)),
                        )

      fig1.update_layout({
                  'title': {'text':'Loss Vs Number of Iterations','x': 0.5,'y': 0.9},
                  'xaxis': {'title': {'text': 'Iterations'}},
                  'yaxis': {'title': {'text': 'Loss'}},
                  })
      
      fig2 = px.scatter(x = np.arange(len(X)), y = np.arange(len(X)))
      fig2.update_layout({'title': {'text':'Regression line','x': 0.5,'y': 0.9}})

      fig3 = px.scatter(x = np.arange(len(X)), y = np.arange(len(X)))
      fig3.update_layout({'title': {'text':'Plots of Thetas Vs Loss','x': 0.5,'y': 0.9},
                         'xaxis': {'title': {'text': 'Theta'}},
                         'yaxis': {'title': {'text': 'Loss'}},})
      
      
      return fig1,fig2, fig3,f'the score is ـــ %' ,f'Gradient Descent converged after ــــ epochs', 0

    
@app.callback(
  Output(component_id = 'visualization feature', component_property = 'figure'),
  Input(component_id = 'button-example-2', component_property = 'n_clicks'),
  State(component_id = 'demo-dropdown1', component_property='value'),
  State(component_id = 'demo-dropdown2', component_property='value'),
  )

def udpate(Button, column, chart):

    if column == None or chart == None:
      fig2 = px.scatter(x = np.arange(len(X)), y = np.arange(len(X)))

    else:
      if chart == 'Scatter' :
          fig2 = px.scatter(x = np.arange(len(X)), y = data[column])

      else : 
          fig2 = ff.create_distplot([data[column]], ['distplot'])

    return fig2     
    
    
    
    
# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ
#  BGFS CallBacks
# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ
  
@app.callback(
    Output('BFGS_Loss interval', 'disabled'),
    Input('BFGS_our-power-button-1', 'on'),
    State('BFGS_Loss interval', 'disabled')
)
def toggle_interval(n_clicks, interval_disabled):
    if n_clicks:
        return False
    else:
        return True
# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ

# Reset the interval index to 0 after power button = false 
@app.callback(
        Output('BFGS_Loss interval', 'n_intervals'),
        Input('BFGS_our-power-button-1', 'on'),
              )
def update_dashboard(Button):
    if ~Button :
        return 0 
# ـــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ

@app.callback(
    Output('BFGS_Loss', 'figure'),
    Output(component_id='BFGS_Regression_line', component_property = 'figure'),
    Output(component_id='BFGS_Theta Graph', component_property = 'figure'),
    Output(component_id='BFGS_score', component_property = 'children'),
    Output(component_id='BFGS_epochs', component_property = 'children'), 
    Output('BFGS_Loss interval', 'max_intervals'),
    Input('BFGS_Loss interval', 'n_intervals'),
    Input(component_id = 'BFGS_our-power-button-1', component_property = 'on'),
    State(component_id = 'BFGS_iteration', component_property = 'value'),
)
def update_graph(n1, Button, iteration):
  if Button:
      BFGS_y_pred, BFGS_all_theta, BFGS_Loss, It = BFGS(X, y, iteration)
      fig1 = px.line(BFGS_Loss[:n1])

      fig1.update_layout({
                      'title': {'text':'Loss Vs Number of Iterations','x': 0.5,'y': 0.9},
                      'xaxis': {'title': {'text': 'Iterations'}},
                      'yaxis': {'title': {'text': 'Loss'}},
                      }) 
      fig2 = go.Figure()
      fig2.add_trace(go.Scatter(x = X.reshape(-1), y = y.reshape(-1), mode='markers', name = 'Data'))
      fig2.add_trace(go.Scatter(x = X.reshape(-1), y = BFGS_y_pred[n1].reshape(-1,1).reshape(-1), name = 'Regression Line'))
      fig2.update_layout({'title': {'text':'Regression Line','x': 0.5,'y': 0.9}})

      fig3 = make_subplots(rows=2, cols=1,
                          subplot_titles=['theta0','theta1'])

      fig3.add_trace(go.Scatter(x = BFGS_all_theta[:n1][:, 0], y = BFGS_Loss, name = 'theta0'), row = 1, col = 1)
      fig3.add_trace(go.Scatter(x = BFGS_all_theta[:n1][:, 1], y = BFGS_Loss, name = 'theta1'), row = 2, col = 1)

      fig3.update_layout({'title': {'text':'Plots of Thetas Vs Loss','x': 0.5,'y': 0.9}})
      score = np.round((r2_score(y, BFGS_y_pred[n1]) * 100), 2)
       
      return fig1,fig2, fig3, f'the score is {score}%',f'Gradient Descent converged after {n1+1} epochs', It
    
  else:    
      fig1 = px.scatter(x = np.arange(len(X)), y = np.arange(len(X)),
                        )

      fig1.update_layout({
                  'title': {'text':'Loss Vs Number of Iterations','x': 0.5,'y': 0.9},
                  'xaxis': {'title': {'text': 'Iterations'}},
                  'yaxis': {'title': {'text': 'Loss'}},
                  })
      
      fig2 = px.scatter(x = np.arange(len(X)), y = np.arange(len(X)))
      fig2.update_layout({'title': {'text':'Regression line','x': 0.5,'y': 0.9}})

      fig3 = px.scatter(x = np.arange(len(X)), y = np.arange(len(X)))
      fig3.update_layout({'title': {'text':'Plots of Thetas Vs Loss','x': 0.5,'y': 0.9},
                         'xaxis': {'title': {'text': 'Theta'}},
                         'yaxis': {'title': {'text': 'Loss'}},})
      
      
      return fig1,fig2, fig3,f'the score is ـــ %' ,f'Gradient Descent converged after ــــ epochs', 0


@app.callback(
  Output(component_id = 'BFGS_visualization feature', component_property = 'figure'),
  Input(component_id = 'BFGS_button-example-2', component_property = 'n_clicks'),
  State(component_id = 'BFGS_demo-dropdown1', component_property='value'),
  State(component_id = 'BFGS_demo-dropdown2', component_property='value'),
  )

def udpate(Button, column, chart):

    if column == None or chart == None:
      fig2 = px.scatter(x = np.arange(len(X)), y = np.arange(len(X)))

    else:
      if chart == 'Scatter' :
          fig2 = px.scatter(x = np.arange(len(X)), y = data[column])

      else : 
          fig2 = ff.create_distplot([data[column]], ['distplot'])

    return fig2 
  
app.run_server(debug=True)


# In[ ]:




