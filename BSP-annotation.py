import streamlit as st
import streamlit_hotkeys as hotkeys
from streamlit import session_state as ss
import pandas as pd
import numpy as np
import json

from streamlit_plotly_events import plotly_events

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime as dt

st.set_page_config(layout="wide")

annotation_dict = {
    #"Artifact - Missing data": 0,
    #"Artifact - Saturation": 1,
    "Artifact - Loose channel": 2,
    "Suppression": 3,
    "50 Hz artifact": 5,
    "Other Event": 4
}

#Initialize session state variables
if "bucket_idx" not in ss:
    ss["bucket_idx"] = 0
if "max_bucket" not in ss:
    ss["max_bucket"] = False
if "bucket_cnt" not in ss:
    ss["bucket_cnt"] = False
if "annot_df" not in ss:
    ss["annot_df"] = pd.DataFrame(data=None, columns=["begin_idx", "end_idx", "trace", "annot_id", "annot_txt", "neurologist", "comment"])
if "timestamp1" not in ss: #temporäre Variablen, um den Zustand der aktuellen Annotation zu tracken.
    ss["timestamp1"] = None
if "ts1_curve" not in ss: #temporäre Variablen, um den Zustand der aktuellen Annotation zu tracken.
    ss["ts1_curve"] = None
if "timestamp2" not in ss:
    ss["timestamp2"] = None
if "view_ts" not in ss:
    ss["view_ts"] = None
if "setup_dict" not in ss:
    ss["setup_dict"] = {
        "strip_len": 30,
        "plot_height": 450,
        "y_range": 200,
        "smooth": 1
    }
if "fig" not in ss:
    ss["fig"] = None
if "overview" not in ss:
    ss["overview"] = None
if "redraw_fig" not in ss:
    ss["redraw_fig"] = True


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

@st.cache_data
def get_smoothed(data: pd.DataFrame, smooth: int) -> pd.DataFrame:
    """Return cached smoothed EEG columns to avoid recalculating rolling mean on every plot render."""
    # compute only the two series we need and return new DataFrame (cached by streamlit)
    return pd.DataFrame({
        "EEG1": data["EEG1"].rolling(smooth).mean().bfill(),
        "EEG2": data["EEG2"].rolling(smooth).mean().bfill()
    }, index=data.index)

def change_bucket_idx(max_bin, offset):
    ss["redraw_fig"] = True
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
        ss["redraw_fig"] = True

def dec_bucket_idx():
    if st.session_state["bucket_idx"] == 0:
        st.warning('Min bucket count reached. Cannot decrement further.', icon="⚠️")
    else:
        st.session_state["bucket_idx"] -= 1
        ss["redraw_fig"] = True

def get_annot():
    if not st.session_state["annot_select"] or st.session_state["annot_select"] == "Select annotation type...":
        return None
    else:
        return st.session_state["annot_select"]

hotkeys.activate([
    hotkeys.hk("previous", "ArrowLeft"),             
    hotkeys.hk("next", "ArrowRight")
    ])

if hotkeys.pressed("previous"):
    dec_bucket_idx()

if hotkeys.pressed("next"):
    inc_bucket_idx()

@st.cache_data
def build_overview_base(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data[::20].index, y=data["EEG1"][::20]+50, mode="lines", marker=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=data[::20].index, y=data["EEG2"][::20]-50, mode="lines", marker=dict(color='#2ca02c')))
    fig.update_layout(height=150, showlegend=False, yaxis_range=[-200, 200], margin=dict(l=20, r=20, t=0, b=0))
    fig["layout"]["yaxis"]["title"] = "POSITION"
    return fig

def build_overview(data, bins):
    fig = build_overview_base(data)
    x0 = bins[st.session_state["bucket_idx"]] - pd.to_timedelta(1, unit='s')
    try:
        x1 = bins[(st.session_state["bucket_idx"]+1)] + pd.to_timedelta(1, unit='s')
    except IndexError:
        x1 = data.index[-1]
    fig.add_vrect(x0=x0, x1=x1, line_width=3, fillcolor="black", opacity=0.3)                 
    return fig

@st.cache_data
def base_plot(data, bins, bucket_idx):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0)

    # compute window bounds once (datetime-like)
    lower_end = bins[bucket_idx] - pd.to_timedelta(1, unit='s')
    if st.session_state["max_bucket"] == False:
        higher_end = bins[(st.session_state["bucket_idx"] + 1)] + pd.to_timedelta(1, unit='s')
    else:
        higher_end = None  # means to end

    # get integer slice positions (faster than label slicing)
    start_pos = data.index.searchsorted(lower_end)
    end_pos = data.index.searchsorted(higher_end) if higher_end is not None else len(data) - 1
    if end_pos < start_pos:
        end_pos = start_pos

    window_slice = slice(start_pos, end_pos + 1)
    window_index = data.index[window_slice]
    eeg1_win = list(data_s["EEG1"].values[window_slice])
    eeg2_win = list(data_s["EEG2"].values[window_slice])

    xs = window_index

    # add traces
    fig.add_trace(go.Scatter(x=xs, y=eeg1_win, mode="lines", line=dict(color='#1f77b4')), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=eeg2_win, mode="lines", line=dict(color='#2ca02c')), row=2, col=1) #"lines+markers"

    return fig, lower_end, higher_end

def build_plot(data, bins, bucket_idx):
    
    fig, lower_end, higher_end = base_plot(data, bins, bucket_idx)

    # vectorized annotation selection (faster than iterrows over whole df)
    if not st.session_state["annot_df"].empty:
        adf = st.session_state["annot_df"]
        # ensure begin_idx/end_idx are datetimes
        # build mask for annotations that overlap window (begin within window)
        mask = (adf["begin_idx"] >= lower_end) & (adf["begin_idx"] <= (higher_end if higher_end is not None else data.index[-1]))
        for annot in adf.loc[mask].itertuples(index=False):
            # access by column names in tuple order: begin_idx, end_idx, trace, annot_id, annot_txt, neurologist, comment
            a_begin, a_end, a_trace, a_id, a_txt, _, _ = annot
            if a_trace == 3:
                fig.add_vrect(x0=a_begin, x1=a_end, line_width=0, fillcolor="red", opacity=0.2)
                fig.add_annotation(x=a_begin, y=-20, text=a_txt, showarrow=False)
            else:
                # row index: trace 1 -> row=1, trace 2 -> row=2
                row_idx = int(a_trace) if a_trace in (1, 2) else 1
                fig.add_vrect(x0=a_begin, x1=a_end, line_width=0, fillcolor="red", opacity=0.2, row=row_idx)
                fig.add_annotation(x=a_begin, y=-20, text=a_txt, showarrow=False, row=row_idx, col=1)

    # plot current annotation timestamps if inside window
    curr_annot = []
    if st.session_state["timestamp1"]:
        curr_annot.append(st.session_state["timestamp1"])
    if st.session_state["timestamp2"]:
        curr_annot.append(st.session_state["timestamp2"])

    for annot_ts in curr_annot:
        if annot_ts is None:
            continue
        if (annot_ts >= lower_end) and (higher_end is None or annot_ts <= higher_end):
            # ts1_curve stored as 0-based curve index, add 1 for subplot row
            row_for_vline = (ss.ts1_curve + 1) if ss.ts1_curve is not None else 1
            fig.add_vline(x=annot_ts, row=row_for_vline)

    ylow, yhigh = ss["setup_dict"]["y_range"] * -1, ss["setup_dict"]["y_range"]
    fig.update_layout(height=ss["setup_dict"]["plot_height"],
                      showlegend=False,
                      yaxis_range=[ylow, yhigh],
                      yaxis2_range=[ylow, yhigh],
                      margin=dict(l=50, r=20, t=20, b=30))
    #fig.update_traces(marker=dict(size=0.01))  # markers tiny if present
    fig.update_xaxes(showgrid=True, tickformat="%S", dtick=1000)
    fig["layout"]["yaxis"]["title"] = "EEG1"
    fig["layout"]["yaxis2"]["title"] = "EEG2"
    return fig

def add_annot(trace_select: str):
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
    
    if trace_select == "single":
        st.session_state["annot_df"].loc[st.session_state["annot_df"].shape[0]] = [st.session_state["timestamp1"], 
                                                                                   st.session_state["timestamp2"], 
                                                                                   ss.ts1_curve+1, 
                                                                                   annotation_dict[st.session_state["valid_annot"]], 
                                                                                   st.session_state["valid_annot"], 
                                                                                   neurologist, 
                                                                                   st.session_state["annot_comment"]]
    elif trace_select == "both":
        st.session_state["annot_df"].loc[st.session_state["annot_df"].shape[0]] = [st.session_state["timestamp1"], 
                                                                                   st.session_state["timestamp2"], 
                                                                                   3, 
                                                                                   annotation_dict[st.session_state["valid_annot"]], 
                                                                                   st.session_state["valid_annot"], 
                                                                                   neurologist, 
                                                                                   st.session_state["annot_comment"]]
    
    #reset ss vars
    st.session_state["timestamp1"] = None
    ss.ts1_curve = None
    st.session_state["timestamp2"] = None
    st.session_state["annot_select"] = "Select annotation type..."
    st.session_state["annot_comment"] = None
    ss["redraw_fig"] = True
    return

def apply_setup():
    if uploaded_file:
        ss["view_ts"] = bins_dt[ss["bucket_idx"]]
    ss["setup_dict"]["strip_len"] = strip_len
    ss["setup_dict"]["plot_height"] = height_slider
    ss["setup_dict"]["y_range"] = y_range
    ss["setup_dict"]["smooth"] = smooth
    ss["redraw_fig"] = True

def new_upload():
    st.session_state["bucket_idx"] = 0
    st.session_state["max_bucket"] = False
    st.session_state["bucket_cnt"] = False
    st.session_state["annot_df"] = pd.DataFrame(data=None, columns=["begin_idx", "end_idx", "trace", "annot_id", "annot_txt", "neurologist", "comment"])
    st.session_state["timestamp1"] = None
    st.session_state["timestamp2"] = None
    st.session_state["view_ts"] = None

def delete_last_annot():
    ss["redraw_fig"] = True
    if not st.session_state["annot_df"].empty:
        st.session_state["annot_df"].drop(st.session_state["annot_df"].index[-1], inplace=True)
    else:
        st.warning('No annotations to delete.', icon="⚠️")

def import_setup(setup_import):
    st.session_state["setup_dict"] = json.load(setup_import)
    ss.strip_len = ss.setup_dict["strip_len"]
    ss.plot_height = ss.setup_dict["plot_height"]
    ss.y_range = ss.setup_dict["y_range"]
    ss.smooth = ss.setup_dict["smooth"]   

def import_annotation(annot_import):
    ss.annot_df = pd.read_csv(annot_import, parse_dates=["begin_idx", "end_idx"], usecols=["begin_idx", "end_idx", "trace", "annot_id", "annot_txt", "neurologist", "comment"])

#data, bins, bins_dt = generate_random_dataset(4375)
with st.sidebar:
    st.title('Burst Suppression Annotation Program')
    neurologist = st.radio("Select annotating neurologist:", ["LM", "KE", "JD"], key="neurologist", horizontal=True)
    uploaded_file = st.file_uploader("Choose a EEG recording file", key="uploader", on_change=new_upload)
    st.divider()
    st.markdown("""## Setup options:""")
    strip_len = st.slider("Select length of EEG plot in s:", key="strip_len", min_value=5, max_value=60, value=30, step=1)
    height_slider = st.slider("Select height of EEG Plot:", key="plot_height", min_value=250, max_value=900, value=450, step=10)
    y_range = st.radio("Set y-axis range (default ± 200µV):", [50, 100, 150, 200], key="y_range", index=3)
    smooth = st.slider("Set moving average smoothing:", 1, 6, 1, 1, key="smooth")
    st.button("Apply changes", key="apply_setup_btn", on_click=apply_setup, use_container_width=True)
    st.divider()
    st.markdown("""## Setup Import/Export:""")
    st.download_button("Export Setup", data=json.dumps(st.session_state["setup_dict"]), 
                           file_name=f"BSP-annotation_setup_{neurologist}.json", use_container_width=True)
    setup_import = st.file_uploader("Import Setup", key="setup_import", type=["json"])
    st.button("Import Setup", key="import_setup_btn", on_click=import_setup, args=(setup_import,), use_container_width=True)
    st.divider()
    st.markdown("""## Artifact types:""")
    st.markdown("""**Missing data**: No Data.  
                **Saturation**: Extrem values (+- 187.5 uV) due to manipulation, movement etc.  
                **Loose channel**: Improbably low variance in the signal due to improperly attached electrode or too noisy signal due to high impedance.""")
    
    

def header_logic():
    st.subheader(f"""{uploaded_file.name.split('.')[0]}: Plotted {ss["setup_dict"]["strip_len"]}s window ({st.session_state["bucket_idx"]+1}/{len(bins)})""")
    if st.session_state["max_bucket"] == False:
        st.write(f"""from {bins_dt[st.session_state["bucket_idx"]].strftime("%H:%M:%S")} to {bins_dt[st.session_state["bucket_idx"]+1].strftime("%H:%M:%S")}""")
    else:
        st.write(f"""from {bins_dt[st.session_state["bucket_idx"]].strftime("%H:%M:%S")} to end of record""")

def control_logic():
    # Control logic
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.button("first", on_click=change_bucket_idx, args=(len(bins)-1, "first"), use_container_width=True)
    with col2:
        st.button("previous", on_click=dec_bucket_idx, use_container_width=True)
    with col3:
        st.button("find Begin Index", on_click=change_bucket_idx, args=(len(bins)-1, "ts"), use_container_width=True)
    with col4:
        st.button("next", on_click=inc_bucket_idx, use_container_width=True)
    with col5:
        st.button("last", on_click=change_bucket_idx, args=(len(bins)-1, "last"), use_container_width=True)    

def annotation_logic():
    global begin_idx_ph, end_idx_ph
    col1, col2, col3 = st.columns(3)
    with col1:
        st.text("Begin Index")
        begin_idx_ph = st.empty()
        st.text("End Index")
        end_idx_ph = st.empty()
        
    with col2:
        optionslist = ["Select annotation type..."] + list(annotation_dict.keys())
        annot_type= st.radio("Select type of annotation:", 
                                optionslist, key="annot_select",
                                index=0) #, placeholder="Select annotation type...",
        annot_comment = st.text_input("Comment (Other Event) annotation:", key="annot_comment")
    with col3:
        st.button("Add **Single Trace** Annotation", key="add_annot_single_bn", on_click=add_annot, use_container_width=True, args=("single",))
        st.button("Add **Both Traces** Annotation", key="add_annot_both_bn", on_click=add_annot, use_container_width=True, args=("both",))
        #streamlit_shortcuts.button("add Annotation", shortcut="Enter", on_click=add_annot)

    with st.expander("Annotations"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(st.session_state["annot_df"])
        with col2:
            st.download_button("**Download Annotations**", data=st.session_state["annot_df"].to_csv(), 
                           file_name=f"{uploaded_file.name.split('.')[0]}_annot_{neurologist}.csv", use_container_width=True)
            st.divider()
            st.button("Delete last Annotation", on_click=delete_last_annot, use_container_width=True)
            st.divider()
            annot_import = st.file_uploader("Import Annotations", key="annot_import", type=["csv"])
            st.button("Import Annotations", key="import_annot_btn", on_click=import_annotation, args=(annot_import,), use_container_width=True)

if uploaded_file is not None:
    if uploaded_file.name.split(".")[0] == "segments":
        ss["setup_dict"]["strip_len"] = 15
    data, bins, bins_dt = prep_data(uploaded_file, ss["setup_dict"]["strip_len"])
    data_s = get_smoothed(data, ss["setup_dict"]["smooth"])

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
    header_logic()
    PlotPlaceholder = st.empty()
    OverviewPlaceholder = st.empty()
    control_logic()
    annotation_logic()

    if ss["redraw_fig"]:
        ss["fig"] = build_plot(data_s, bins, st.session_state["bucket_idx"])
        ss["overview"] = build_overview(data, bins)
        ss["redraw_fig"] = False
    
    with PlotPlaceholder.container():
        clickedPoint = plotly_events(ss["fig"], click_event=True, override_height=ss["setup_dict"]["plot_height"]) #
        #st.plotly_chart(fig, use_container_width=True, on_select=point_clicked, key="click_data", config = {'displayModeBar': False} )
        
    
    with OverviewPlaceholder.container():
        st.plotly_chart(ss["overview"], use_container_width=True, key="overview", config = {'staticPlot': True} )
        
    if clickedPoint:
        curve = clickedPoint[0]["curveNumber"]
        ts = dt.strptime(clickedPoint[0]["x"], "%Y-%m-%d %H:%M:%S.%f")
        valid_annot = get_annot()
        st.session_state["valid_annot"] = valid_annot
        #import os
        #os.write(1,b'Klick registered.\n')
        #print("Klick registered.", ts, curve)
        if not st.session_state["timestamp1"]: #first click in plot
            st.session_state["timestamp1"] = ts
            ss.ts1_curve = curve
            
        elif st.session_state["timestamp1"] and not valid_annot:
            st.session_state["timestamp1"] = ts
            ss.ts1_curve = curve
        
        elif st.session_state["timestamp1"] and valid_annot and not st.session_state["timestamp2"]: #subsequent clicks in plot != previous point
            st.session_state["timestamp2"] = ts
        
        elif st.session_state["timestamp1"] and valid_annot and st.session_state["timestamp2"]:
            st.session_state["timestamp2"] = ts

        ss["fig"] = build_plot(data_s, bins, st.session_state["bucket_idx"])
        with PlotPlaceholder.container():
            clickedPoint = plotly_events(ss["fig"], click_event=True, override_height=ss["setup_dict"]["plot_height"])
        if st.session_state["timestamp1"]:
            begin_idx_ph.write(st.session_state["timestamp1"])
        if st.session_state["timestamp2"]:
            end_idx_ph.write(st.session_state["timestamp2"])

else:
    st.write("Please upload file to start annotating!")















