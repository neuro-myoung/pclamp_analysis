import streamlit as st
import re
import numpy as np
import pandas as pd
import io
import cufflinks as cf 
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px
from scipy.optimize import curve_fit
import base64


############################################## Define Functions ####################################################
@st.cache(allow_output_mutation=True)
def load_file(path):
    '''
    This function will parse a standard HEKA .asc file into a pandas dataframe.

    Arguments: 
    path - a stringIO input of a standard HEKA output .asc file.

    Returns:
    df, dfcache - two copies of the file reformatted into a dataframe.
    '''

    lineIndices = []                                      
    rawFile = path.getvalue().strip().split("\n")         # Splits string at \n and removes trailing spaces

    count=0                                               
    for line in rawFile:                                  # Finds rows that contain header information to exclude from df
        if re.search(r"[a-z]+", line) == None:           
            lineIndices.append(count)                     
        count += 1                                    
    
    processedFile = [rawFile[i].strip().replace(" ", "").split(",") for i in lineIndices]     # Formats headerless file for later df

    nSweeps = int((len(rawFile)-len(processedFile)-1)/2)   # Use the difference in file with and without headers to find nSweeps

    if len(processedFile[0]) == 5:
        colnames = ['index','ti','i','tp','p']
    else:
        colnames = ['index','ti','i','tp','p','tv','v']

    df = pd.DataFrame(columns=colnames, data=processedFile)
    df = df.apply(pd.to_numeric)
    df = df.dropna(axis=0)
    df['sweep'] = np.repeat(np.arange(nSweeps), len(df)/nSweeps)
    
    # Change units to something easier to work with
    df['p'] = df['p'] / 0.02
    df['ti'] *= 1000
    df['i'] *= 1e12
    df['tp'] *= 1000
    df_cache = df.copy()

    return df, df_cache

@st.cache(allow_output_mutation=True)
def plot_sweeps(df):
    '''
    This function will plot a dataframe of sweeps using plotly with hidden axis.

    Arguments: 
    df - a dataframe with columns tp, p, ti, i, and sweep

    Returns:
    fig - a plotly figure object
    '''

    fig = make_subplots(rows=2, cols=1,  row_width=[0.6, 0.3])
    
    for name, sweep in df.groupby('sweep'):
        
        fig.add_trace(
            go.Scatter(mode='lines', name=name, x=sweep.tp, y=sweep.p, marker=dict(color='#800000'),
                hovertemplate='x: %{x}<br>' + 'y: %{y}<br>'),
            row=1, col=1)
            
        fig.add_trace(
            go.Scatter(mode='lines', name=name, x=sweep.ti, y=sweep.i, marker=dict(color='black'),
                hovertemplate='x: %{x}<br>' + 'y: %{y}<br>'),
            row=2, col=1)

    fig.update_layout(
        height=600,
        width=800,
        template='none',
        xaxis_showticklabels=False,
        xaxis_showgrid=False,
        yaxis_showticklabels=False,
        yaxis_showgrid=False,
        xaxis2_showticklabels=False,
        xaxis2_showgrid=False,
        yaxis2_showticklabels=False,
        yaxis2_showgrid=False,
        showlegend=False,
        hovermode='closest')

    fig.update_xaxes(matches='x')

    return(fig)

def highlight_fig(fig, window, draw=False):
    '''
    This function will highlight a selected region of the plot.

    Arguments: 
    fig - a plotly figure object.
    window - an iterable with the start and end of the selection window.
    draw - boolean indicating whether or not to draw a highlight (default is False).
    Returns:
    fig - a plotly figure object
    '''

    if draw == False:
        highlight_color = "white"
    else:
        highlight_color = "LightSalmon"
    fig.update_layout(
        shapes=[
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=window[0],
                y0=0,
                x1=window[1],
                y1=1,
                fillcolor=highlight_color,
                opacity=0.5,
                layer="below",
                line_width=0,
                )
            ]
        )
    return(fig)
        
def add_scalebars(df, fig, locs):
    '''
    This function will add scalebars to a plot.

    Arguments: 
    df - a pandas dataframe with columns p, ti, tp, and i.
    fig - a plotly figure object..
    locs - a dictionary with the axis names as keys and scalebar limits as values.

    Returns:
    fig - a plotly figure object
    '''

    try:
        if all(value == 0 for value in locs['p']) == False:
            pscale = dict(type="line", 
                        x0=locs['t'][0],
                        x1=locs['t'][0], 
                        y0=locs['p'][0], 
                        y1=locs['p'][1],
                        line=dict(color="black",
                                    width=2))

            fig.add_shape(pscale, row=1, col=1)

        if all(value == 0 for value in locs['i']) == False:
            iscale = dict(type="line", 
                        x0=locs['t'][0], 
                        x1=locs['t'][0], 
                        y0=locs['i'][0], 
                        y1=locs['i'][1],
                        line=dict(color="black",
                                    width=2))

            fig.add_shape(iscale, row=2, col=1)
            
        if all(value == 0 for value in locs['t']) == False:
            tscale = dict(type="line", 
                        x0=locs['t'][0], 
                        x1=locs['t'][1], 
                        y0=locs['i'][0], 
                        y1=locs['i'][0],
                        line=dict(color="black",
                                    width=2))
            
            fig.add_shape(tscale, row=2, col=1)
    except (KeyError, TypeError):
        print("Values must be entered as space separated integers.")   
    return(fig) 
    
def baseline_subtract(df, window):
    '''
    This function will baseline subtract a dataframe based on a given window.

    Arguments: 
    df - a pandas dataframe with columns p, ti, tp, and i.
    window - an iterable with the start and end coordinates of the baseline window.
    
    Returns:
    df - a modified pandas dataframe.
    '''

    iblsub = []
    grouped = df.groupby('sweep')
    baselines = df.query('ti >= @window[0] and ti < @window[1]').groupby('sweep')['i'].mean()
                
    for name,group in grouped['i']:
        iblsub.append(group-baselines[name])
        
    flatList = [item for sublist in iblsub for item in sublist]
    df['i'] = flatList
    
    return(df)

@st.cache
def sweep_summary(df, window, param):
    '''
    This function will summarize sweep data based on a selected summary statistic.

    Arguments: 
    df - a pandas dataframe with columns p, ti, tp, sweep, and i.
    window - an iterable with the start and end coordinates of the baseline window.
    param - a summary statistic by which to summarize the data ('Max', 'Min' or 'Mean' currently accepted).
    
    Returns:
    df - a dataframe of summary data by sweep.
    '''

    subsetDf = df.query('ti >= @window[0] and ti < @window[1]')
    groups = subsetDf.groupby('sweep')

    if param == 'None':
        return
    elif param == 'Mean':
        iMean = groups['i'].mean()
        summaryDict = {
            'pressure': np.abs(groups['p'].median()),
            'mean_i': iMean,
            'mean_norm_i': np.abs(iMean)/np.max(np.abs(iMean)),
            'stdev_i': groups['i'].std()
        }

        summaryDf = pd.DataFrame(summaryDict)

    elif param == 'Min':
        iMin = groups['i'].min()
        summaryDict = {
            'pressure': np.abs(groups['p'].median()),
            'min_i': iMin,
            'min_norm_i': iMin/np.min(iMin)
        }

        summaryDf = pd.DataFrame(summaryDict)
    else:
        iMax = groups['i'].max()
        summaryDict = {
            'pressure': np.abs(groups['p'].median()),
            'max_i': iMax,
            'max_norm_i': iMax/np.max(iMax)
        }

        summaryDf = pd.DataFrame(summaryDict)
    return summaryDf

def plot_summary(df, yval):
    '''
    This function will plot a dataframe of summary statistics as a function of stimulus intensity.

    Arguments: 
    df - a pandas dataframe with columns pressure, param, and normalized_param.
    window - an iterable with the start and end coordinates of the baseline window.
    
    Returns:
    df - a plotly figure object.
    '''

    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(mode='markers',
                   name='p50', 
                   marker_color='#FF3300', 
                   marker_line_width = 1,
                   marker_size = 10,
                   x=df['pressure'], 
                   y=df[yval],
                   hovertemplate='x: %{x}<br>' + 'y: %{y}<br>'
                   )
    )

    fig.update_xaxes(title_text='Pressure (-mm Hg)')
    fig.update_yaxes(title_text='I/Imax')

    fig.update_layout(
        height=600,
        width=800,
        template='simple_white',
        showlegend=False,
        hovermode='closest')

    return(fig)

def fit_layer(df, fig, fit):
    '''
    This function plots fit data over an existing plot.

    Arguments: 
    df - a pandas dataframe with columns pressure, param, and normalized_param.
    fig - a plotly figure object.
    fit - the fit parameters for a sigmoid fit.
    
    Returns:
    df - a plotly figure object.
    '''

    xfine = np.linspace(min(df.pressure),max(df.pressure), 100)
    fig.add_trace(
    go.Scatter(mode='lines',
               name='fit', 
               marker_color='black', 
               marker_line_width = 1,
               x=xfine, 
               y=sigmoid_fit(xfine, *popt),
               hovertemplate='x: %{x}<br>' + 'y: %{y}<br>'
               )
    )

    return(fig)

def sigmoid_fit(p, p50, k):
    '''
    This function defines a sigmoid curve.

    Arguments: 
    p - the abscissa data.
    p50 - the inflection point of the sigmoid.
    k - the slope at the inflection point of a sigmoid.
    
    Returns:
    The ordinate for a boltzmann sigmoid with the passed parameters.
    '''

    return(1 / (1 + np.exp((p50 - p) / k)))

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
    return(href)

def linear_fit(x, m, b):
    y = m*x + b
    return(y);k
    
#####################################################################################################################

st.beta_set_page_config(
    page_title="Ephys Analysis",
    page_icon="🧊")
    
st.set_option('deprecation.showfileUploaderEncoding', False)
st.sidebar.subheader('Data:')
st.sidebar.markdown("Drag and drop HEKA .asc export files. Keep table display at 'None' when not in use for performance.")
data = st.sidebar.file_uploader("", type=["csv", "asc", "txt"])

## Display a warning and stop run if there is no data uploaded
if data is None:
    st.warning('Upload a file to run the app...')
    st.stop()

########################################### Sweep View ########################################################

st.title('Pressure Clamp Analysis')
st.header('1. Sweep View')

[df, df_cache] = load_file(data)
df = filter_data(df)
# Recover original dataframe
if st.sidebar.button('Reset Data'):
    df = df_cache

tableDisplay = st.sidebar.selectbox(
    'Table Display:',
    ('None', 'Head', 'Tail', 'All'))

if tableDisplay == 'Head':
    st.dataframe(df.head(10))
elif tableDisplay == 'Tail':
    st.dataframe(df.tail(10))
elif tableDisplay == 'All':
    st.dataframe(df)

st.sidebar.markdown('---')
st.sidebar.header('Preprocessing:')
st.sidebar.markdown('Use the slider or text boxes to set windows for baseline subtract or subsetting data.')

default_window = (min(df.ti), max(df.ti))
window = default_window 
    
plot_area = st.empty()

# Preprocessing
baselineWindow = st.sidebar.slider('Baseline Window:', min(df.ti), max(df.ti), (min(df.ti), max(df.ti)), step=1.)

if st.sidebar.button('Baseline Subtract'):
    df = baseline_subtract(df, baselineWindow)

subsetWindow = st.sidebar.slider('Subset Window:', min(df.ti), max(df.ti), (min(df.ti), max(df.ti)), step=1.)

if st.sidebar.button('Subset Data'):
    df = df.query('ti >= @subsetWindow[0] and ti < @subsetWindow[1]')

sweepIDs = list(range(0, len(np.unique(df.sweep))))
removeList = st.sidebar.multiselect(
    'Select sweeps to remove if any:',
    sweepIDs)

if st.sidebar.button('Remove Sweep:'):
    df = df.query('sweep not in @removeList')

if subsetWindow != window:
    window = subsetWindow
elif (subsetWindow == window) & (baselineWindow != window):
    window = baselineWindow
else:
    window = default_window

fig = plot_sweeps(df)

st.sidebar.markdown('---')

# Scalebar
st.subheader('Scalebars')    
st.markdown("For scalebars add integer start and end points as space separated values.")
tScaleInput = st.text_input("Time axis (ms)", "0 0")
pScaleInput = st.text_input("Pressure axis (mm Hg)", "0 0")
iScaleInput = st.text_input("Current axis (pA)", "0 0")

tbars = [int(i) for i in tScaleInput.split()]
pbars = [int(i) for i in pScaleInput.split()]
ibars = [int(i) for i in iScaleInput.split()]

scalebarDict = {'t':tbars, 'i':ibars, 'p':pbars}

if st.button('Add Scalebars'):
    fig = add_scalebars(df, fig, scalebarDict)

st.markdown('---')

###########################################################################################################
############################################### P50 Analysis ##############################################

st.sidebar.header('P50 Analysis:')
st.sidebar.markdown('Select a plot region and a summary parameter to extract these values from each sweep.')
selectWindow = st.sidebar.slider('Select Plot Region:', min(df.ti), max(df.ti), (min(df.ti), max(df.ti)), step=1.)

if (subsetWindow == default_window) & (selectWindow != default_window):
    window = selectWindow

if window != default_window:
    fig = highlight_fig(fig, window, draw=True)
else:
    fig = highlight_fig(fig, window, draw=False)

# Plot Sweeps
plot_area.plotly_chart(fig)

summaryParam = st.sidebar.selectbox(
    'Summary Parameter:',
    ('Max', 'Min', 'Mean'))

st.header('2. P50 Analysis')

# Summarize data and subsequent
agree = st.sidebar.checkbox("Summarize Data:")

# Stimulus response analysis
if agree:
    summaryDf = sweep_summary(df, selectWindow, summaryParam)
    st.dataframe(summaryDf)

    if 'mean_i' in summaryDf.columns.values:
        yval = 'mean_norm_i'
    elif 'min_i' in summaryDf.columns.values:
        yval = 'min_norm_i'
    else:
        yval = 'max_norm_i'

    aggFig = plot_summary(summaryDf, yval)

    st.sidebar.markdown('Fitting Parameters')
    summaryWindowDefault = (min(summaryDf.pressure), max(summaryDf.pressure))
    summaryWindow = summaryWindowDefault 
    fitWindow = st.sidebar.slider('Select Plot Region:', 
                                  min(summaryDf.pressure), max(summaryDf.pressure), 
                                  (min(summaryDf.pressure), max(summaryDf.pressure)), step=1.)
    
    if fitWindow != summaryWindow:
        highlight_fig(aggFig, fitWindow, draw=True)
    else:
        highlight_fig(aggFig, fitWindow, draw=False)

    # Fit a sigmoid function and plot over the data.
    if st.sidebar.button('Fit Sigmoid'):
        summaryWindow = summaryWindowDefault
        popt, pcov = curve_fit(sigmoid_fit, summaryDf.pressure, summaryDf[yval])
        plotFit = popt
        aggFig = fit_layer(summaryDf, aggFig, popt)
        st.plotly_chart(aggFig)
        st.write('P50: ' + str(popt[0]))
        st.write('slope: ' + str(popt[1]))
    else:
        summaryWindow = fitWindow
        st.plotly_chart(aggFig)

    st.sidebar.markdown('---')

    #Save dataframe as csv
    st.sidebar.markdown('Add a unique identifier and save the file as a csv.')
    uniqueID = st.sidebar.text_input('Unique ID', 'date_cellnumber_protocol')

    if st.sidebar.button('Generate Save File'):
        if "uniqueID" not in summaryDf.columns:
            summaryDf.insert(0, "uniqueID", np.repeat(uniqueID,np.shape(summaryDf)[0]))
        st.sidebar.markdown(get_table_download_link(summaryDf), unsafe_allow_html=True)

