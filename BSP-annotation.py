import streamlit as st
from streamlit_plotly_events import plotly_events
import streamlit_shortcuts
from streamlit import session_state as ss
import pandas as pd
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime as dt
from datetime import timedelta 

st.set_page_config(layout="wide")


st.title('Burst Suppression Annotation Program')

if "bucket_idx" not in st.session_state:
    st.session_state["bucket_idx"] = 0
if "max_bucket" not in st.session_state:
    st.session_state["max_bucket"] = False
if "bucket_cnt" not in st.session_state:
    st.session_state["bucket_cnt"] = False
if "annot_df" not in st.session_state:
    st.session_state["annot_df"] = pd.DataFrame(data=None, columns=["begin_idx", "end_idx", "annot_id", "annot_txt", "neurologist"])
if "timestamp1" not in st.session_state: #temporäre Variablen, um den Zustand der aktuellen Annotation zu tracken.
    st.session_state["timestamp1"] = None
if "timestamp2" not in st.session_state:
    st.session_state["timestamp2"] = None
if "view_ts" not in st.session_state:
    st.session_state["view_ts"] = None

@st.cache_data
def generate_random_dataset(n):
    rand_data = 5 * np.random.randn(n, 2)
    date_range = pd.date_range(start=dt.now(), periods=n, freq="8ms")
    df = pd.DataFrame(data=rand_data, columns=["EEG1", "EEG2"], index=date_range)
    bins = df.asfreq("10s").index.to_list()
    bins_dt = [x.to_pydatetime() for x in bins]
    return df, bins, bins_dt

@st.cache_data
def prep_data(uploaded_file, strip_len):
    df = pd.read_csv(uploaded_file, usecols=["timestamp", "EEG1","EEG2"])
    df['index_col'] = pd.Timestamp('2024-01-01')
    df['index_col'] = df['index_col'] + pd.to_timedelta(df['timestamp'], unit='ms')
    df.set_index('index_col', inplace=True)

    bins = df.asfreq(f"{strip_len}s").index.to_list()
    bins_dt = [x.to_pydatetime() for x in bins]
    #st.write(df)
    return df, bins, bins_dt
    
def change_bucket_idx(max_bin, offset):
    if offset == "first":
        st.session_state["bucket_idx"] = 0
    elif offset == "last":
        st.session_state["bucket_idx"] = max_bin
    elif offset == "next":
        if st.session_state["bucket_idx"] == max_bin:
            st.warning('Max bucket count reached. Cannot increment further.', icon="⚠️")
        else:
            st.session_state["bucket_idx"] += 1
    elif offset == "previous":
        if st.session_state["bucket_idx"] == 0:
            st.warning('Min bucket count reached. Cannot decrement further.', icon="⚠️")
        else:
            st.session_state["bucket_idx"] -= 1
    elif offset == "ts":
        if not st.session_state["timestamp1"]:
            st.warning('Begin Index not set.', icon="⚠️")
        else:
            bucket = min([i for i in bins_dt if i <= st.session_state["timestamp1"]], key=lambda x: abs(x - st.session_state["timestamp1"]))
            st.session_state["bucket_idx"] = bins_dt.index(bucket)
    elif offset == "view_ts":
            bucket = min([i for i in bins_dt if i <= st.session_state["view_ts"]], key=lambda x: abs(x - st.session_state["view_ts"]))
            st.session_state["bucket_idx"] = bins_dt.index(bucket)

def inc_bucket_idx():
    if st.session_state["bucket_idx"] == st.session_state["bucket_cnt"]:
        st.warning('Max bucket count reached. Cannot incement further.', icon="⚠️")
    else:
        st.session_state["bucket_idx"] += 1
def dec_bucket_idx():
    if st.session_state["bucket_idx"] == 0:
        st.warning('Min bucket count reached. Cannot decrement further.', icon="⚠️")
    else:
        st.session_state["bucket_idx"] -= 1

def get_annot():
    if not st.session_state["annot_select"] or st.session_state["annot_select"] == "Select annotation type...":
        return None
    else:
        return st.session_state["annot_select"]

def build_plot(data, bins, bins_dt):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    lower_end = bins[st.session_state["bucket_idx"]]  - pd.to_timedelta(1, unit='s')
    if st.session_state["max_bucket"] == False:
        higher_end = bins[(st.session_state["bucket_idx"]+1)] + pd.to_timedelta(1, unit='s')
        fig.add_trace(go.Scatter(x=data[lower_end:higher_end].index, y=data[lower_end:higher_end]["EEG1"], mode="lines", marker=dict(color='#1f77b4')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data[lower_end:higher_end].index, y=data[lower_end:higher_end]["EEG2"], mode="lines", marker=dict(color='#2ca02c')), row=2, col=1)

    else:
        fig.add_trace(go.Scatter(x=data[lower_end:].index, y=data[lower_end:]["EEG1"], mode="lines", marker=dict(color='#1f77b4')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data[lower_end:].index, y=data[lower_end:]["EEG2"], mode="lines", marker=dict(color='#2ca02c')), row=2, col=1)

    for idx, annot in st.session_state["annot_df"].iterrows():
        low = bins_dt[st.session_state["bucket_idx"]] - timedelta(seconds=1)
        if st.session_state["max_bucket"] == False:
            high = bins_dt[st.session_state["bucket_idx"]+1] + timedelta(seconds=1)
        else:
            high = data.index[-1]
        if (annot[0] >= low) & (annot[0] <= high):
            fig.add_vrect(x0=annot[0], x1=annot[1], line_width=0, fillcolor="red", opacity=0.2)
            fig.add_annotation(x=annot[0], y=-20, text=annot[3], showarrow=False)

    # plot currect annotation
    curr_annot = []
    if st.session_state["timestamp1"]:
        curr_annot.append(st.session_state["timestamp1"])
    if st.session_state["timestamp2"]:
        curr_annot.append(st.session_state["timestamp2"])

    for annot in curr_annot:
        #only plot if within currect window
        low = bins_dt[st.session_state["bucket_idx"]] - timedelta(seconds=1)
        if st.session_state["max_bucket"] == False:
            high = bins_dt[st.session_state["bucket_idx"]+1] + timedelta(seconds=1)
        else:
            high = data.index[-1]
        if (annot >= low) & (annot <= high):
            fig.add_vline(x=annot)

    fig.update_layout(showlegend=False, yaxis_range=[-200,200], yaxis2_range=[-200,200])
    #fig.update_layout(hovermode=False)
    fig["layout"]["yaxis"]["title"] = "EEG1"
    fig["layout"]["yaxis2"]["title"] = "EEG2"
    return fig

def control_logic():
    # Control logic
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.button("first", on_click=change_bucket_idx, args=(len(bins)-1, "first"), use_container_width=True)
    with col2:
        streamlit_shortcuts.button("previous", shortcut="ArrowLeft", on_click=dec_bucket_idx)
    with col3:
        st.button("find Begin Index", on_click=change_bucket_idx, args=(len(bins)-1, "ts"), use_container_width=True)
    with col4:
        streamlit_shortcuts.button("next", shortcut="ArrowRight", on_click=inc_bucket_idx)
    with col5:
        st.button("last", on_click=change_bucket_idx, args=(len(bins)-1, "last"), use_container_width=True)

def add_annot():
    #check for correct variables
    if not st.session_state["timestamp1"]:
        st.warning('Begin timestamp not defined.', icon="⚠️")
        return
    if not st.session_state["valid_annot"]:
        st.warning('No valid annotation selected.', icon="⚠️")
        return
    if not st.session_state["timestamp2"]:
        st.warning('End timestamp not defined.', icon="⚠️")
        return
    if st.session_state["timestamp1"] > st.session_state["timestamp2"]:
        st.warning('Begin timestamp lager than end timestamp, please correct', icon="⚠️")
        return
    
    st.session_state["annot_df"].loc[st.session_state["annot_df"].shape[0]] = [st.session_state["timestamp1"], st.session_state["timestamp2"], annotation_dict[st.session_state["valid_annot"]], st.session_state["valid_annot"], neurologist]
    
    #reset ss vars
    st.session_state["timestamp1"] = None
    st.session_state["timestamp2"] = None
    st.session_state["annot_select"] = "Select annotation type..."
    return

def slider_change():
    if uploaded_file:
        st.session_state["view_ts"] = bins_dt[st.session_state["bucket_idx"]]

def new_upload():
    st.session_state["bucket_idx"] = 0
    st.session_state["max_bucket"] = False
    st.session_state["bucket_cnt"] = False
    st.session_state["annot_df"] = pd.DataFrame(data=None, columns=["begin_idx", "end_idx", "annot_id", "annot_txt", "neurologist"])
    st.session_state["timestamp1"] = None
    st.session_state["timestamp2"] = None
    st.session_state["view_ts"] = None

annotation_dict = {
    "Artifact - Missing data": 0,
    "Artifact - Saturation": 1,
    "Artifact - Loose channel": 2,
    "Suppression": 3
}

#data, bins, bins_dt = generate_random_dataset(4375)
with st.expander("Setup"):
    col1, col2 = st.columns(2)
    with col1:
        neurologist = st.radio("Select annotating neurologist:", ["VM", "JD"], key="neurologist")
        strip_len = st.number_input("Select length of EEG plot in s:", min_value=5, max_value=60, value=30, step=1, on_change=slider_change)
    with col2:
        uploaded_file = st.file_uploader("Choose a EEG recording file", key="uploader", on_change=new_upload)

if uploaded_file is not None:
    if uploaded_file.name.split(".")[0] == "segments":
        strip_len = 15
    data, bins, bins_dt = prep_data(uploaded_file, strip_len)
    
    #Update session state variables
    if st.session_state["bucket_idx"] == len(bins)-1:
        st.session_state["max_bucket"] = True
    else:
        st.session_state["max_bucket"] = False
        st.session_state["bucket_cnt"] = len(bins)-1
    
    #if slider was used center strip on last view
    if st.session_state["view_ts"]:
        change_bucket_idx(len(bins)-1, "view_ts")
        st.session_state["view_ts"] = None

    # Plot Header
    st.subheader(f"""{uploaded_file.name.split('.')[0]}: Plotted {strip_len}s window ({st.session_state["bucket_idx"]+1}/{len(bins)})""")
    if st.session_state["max_bucket"] == False:
        st.write(f"""from {bins_dt[st.session_state["bucket_idx"]].strftime("%H:%M:%S")} to {bins_dt[st.session_state["bucket_idx"]+1].strftime("%H:%M:%S")}""")
    else:
        st.write(f"""from {bins_dt[st.session_state["bucket_idx"]].strftime("%H:%M:%S")} to end of record""")


    control_logic()

    fig = build_plot(data, bins, bins_dt)

    clickedPoint = plotly_events(fig, click_event=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.text("Begin Index")
        begin_idx = st.empty()
        if st.session_state["timestamp1"]:
            begin_idx.write(st.session_state["timestamp1"])
        st.text("End Index")
        end_idx = st.empty()
        if st.session_state["timestamp2"]:
            end_idx.write(st.session_state["timestamp2"])
    with col2:
        optionslist = ["Select annotation type..."] + list(annotation_dict.keys())
        annot_type= st.radio("Select type of annotation:", 
                                optionslist, key="annot_select",
                                index=0) #, placeholder="Select annotation type...",
    with col3:
        streamlit_shortcuts.button("add Annotation", shortcut="Enter", on_click=add_annot)

    with st.expander("Annotations"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(st.session_state["annot_df"])
        with col2:
            st.download_button("Download annotations", data=st.session_state["annot_df"].to_csv(), 
                           file_name=f"{uploaded_file.name.split('.')[0]}_annot_{neurologist}.csv", use_container_width=True)

    #if st.checkbox("Show bins"):
    #    st.write(bins)
    #    st.write(bins_dt)

    #if st.checkbox('Show raw data'):
    #    st.subheader('Raw data')
    #    st.write(data)


    
    if clickedPoint:
        ts = dt.strptime(clickedPoint[0]["x"], "%Y-%m-%d %H:%M:%S.%f")
        valid_annot = get_annot()
        st.session_state["valid_annot"] = valid_annot
        if not st.session_state["timestamp1"]: #first click in plot
            st.session_state["timestamp1"] = ts
            st.rerun()
        elif st.session_state["timestamp1"] and not valid_annot:
            st.session_state["timestamp1"] = ts
            st.rerun()
        elif st.session_state["timestamp1"] and valid_annot and not st.session_state["timestamp2"]: #subsequent clicks in plot != previous point
            st.session_state["timestamp2"] = ts
            st.rerun()
        elif st.session_state["timestamp1"] and valid_annot and st.session_state["timestamp2"]:
            st.session_state["timestamp2"] = ts
            st.rerun()
else:
    st.write("Please upload file to start annotating!")

#ss

with st.expander("Artifact types"):
    st.markdown("""**Missing data**: Flat line or no Data.  
                **Saturation**: Extrem values (+- 187.5 uV) due to manipulation.  
                **Loose channel**: Improbably low variance in the signal due to improperly attached electrode.""")
# Delete a single key-value pair
#del st.session_state[key]

# Delete all the items in Session state
#for key in st.session_state.keys():
#    del st.session_state[key]
