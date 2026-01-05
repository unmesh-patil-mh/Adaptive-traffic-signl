import streamlit as st
import pandas as pd
import time

# Configuration
FIXED_CYCLE_TIME = 65  # Traditional system cycle time
YELLOW_TRANSITION_TIME = 5  # Transition period between phases

# ---------------------------
# Traffic Light Component
# ---------------------------
def create_traffic_light(col, lane_data, phase, timer, transition_alert=False):
    with col:
        with st.container(border=True):
            # Header with alerts
            header_cols = st.columns([4, 1])
            header_cols[0].markdown(f"### {lane_data['Lane'].upper().replace('_', ' ')}")
            
            # Status indicators
            if lane_data['Emergency'].lower() == 'yes':
                header_cols[1].markdown("""<span style='color:red; font-size:24px'>🚨</span>""", 
                                      unsafe_allow_html=True)
            elif transition_alert:
                header_cols[1].markdown("""<span style='color:orange; font-size:24px'>⚠️</span>""", 
                                      unsafe_allow_html=True)
            
            # Light indicators
            light_cols = st.columns(3)
            colors = {
                'green': ('#00ff00' if phase == 'green' else '#004400'),
                'yellow': ('#ffd700' if phase == 'yellow' else '#443300'),
                'red': ('#ff0000' if phase == 'red' else '#440000')
            }
            
            for lcol, (name, color) in zip(light_cols, colors.items()):
                lcol.markdown(f"<div style='height:30px; width:30px; border-radius:50%; background:{color}; margin:auto;'></div>", 
                            unsafe_allow_html=True)
            
            # Timer display
            max_time = (FIXED_CYCLE_TIME if phase == 'red' else 
                       lane_data['Green_Light_Time'] if phase == 'green' else 
                       YELLOW_TRANSITION_TIME)
            progress = timer / max_time if max_time > 0 else 0
            st.progress(min(progress, 1.0))
            
            delta_value = (f"Saved {FIXED_CYCLE_TIME - lane_data['Green_Light_Time']}s" 
                          if phase == 'green' else None)
            st.metric("Time Remaining", f"{max(0, timer)}s", delta=delta_value)
            
            st.caption(f"""
            **Live Traffic**
            🚗 {lane_data['Cars']} | 🏍 {lane_data['Motorcycle']}
            🚚 {lane_data['Trucks_Buses']} | 🚑 {lane_data['Ambulance']}
            """)

# ---------------------------
# Simulation Logic
# ---------------------------
def run_synchronized_simulation(data):
    lanes = data.to_dict('records')
    total_saved = 0
    cumulative_saving = 0
    start_time = time.time()
    
    # Initialize expected waiting times
    base_waiting = {lane['Lane']: i*FIXED_CYCLE_TIME for i, lane in enumerate(lanes)}
    
    # Create display columns
    cols = st.columns(len(lanes))
    placeholders = {lane['Lane']: col.empty() for lane, col in zip(lanes, cols)}
    
    try:
        for current_idx, current_lane in enumerate(lanes):
            # Calculate phase parameters
            green_time = current_lane['Green_Light_Time']
            saved_time = FIXED_CYCLE_TIME - green_time
            
            # GREEN PHASE
            phase_start = time.time()
            while (time.time() - phase_start) < green_time:
                total_elapsed = time.time() - start_time
                
                # Update current lane
                remaining_green = green_time - (time.time() - phase_start)
                create_traffic_light(
                    placeholders[current_lane['Lane']],
                    current_lane,
                    'green',
                    int(remaining_green)
                )
                
                # Update other lanes
                for lane in lanes:
                    if lane['Lane'] != current_lane['Lane']:
                        expected_wait = base_waiting[lane['Lane']] - cumulative_saving
                        remaining_wait = max(0, expected_wait - total_elapsed)
                        create_traffic_light(
                            placeholders[lane['Lane']],
                            lane,
                            'red',
                            int(remaining_wait))
                
                time.sleep(0.1)
            
            # YELLOW TRANSITION
            phase_start = time.time()
            while (time.time() - phase_start) < YELLOW_TRANSITION_TIME:
                total_elapsed = time.time() - start_time
                
                # Update current lane
                remaining_yellow = YELLOW_TRANSITION_TIME - (time.time() - phase_start)
                create_traffic_light(
                    placeholders[current_lane['Lane']],
                    current_lane,
                    'yellow',
                    int(remaining_yellow),
                    transition_alert=True)
                
                # Update other lanes (without applying current savings yet)
                for lane in lanes:
                    if lane['Lane'] != current_lane['Lane']:
                        expected_wait = base_waiting[lane['Lane']] - cumulative_saving
                        remaining_wait = max(0, expected_wait - total_elapsed)
                        create_traffic_light(
                            placeholders[lane['Lane']],
                            lane,
                            'red',
                            int(remaining_wait))
                
                time.sleep(0.1)
            
            # Update cumulative savings AFTER yellow transition
            cumulative_saving += saved_time
            total_saved += saved_time
            
            # Reset previous lane to red
            create_traffic_light(
                placeholders[current_lane['Lane']],
                current_lane,
                'red',
                0
            )
    
    except KeyboardInterrupt:
        st.stop()
    
    st.success(f"Total Time Saved This Cycle: {total_saved} seconds")

# ---------------------------
# Main Interface
# ---------------------------
st.set_page_config(page_title="Smart Traffic Control", layout="wide")
st.title("🚦 Adaptive Traffic Signal System")

# Load data
df = pd.read_csv("final_lane_output.csv")

# Configuration panel
with st.expander("⚙️ System Configuration", expanded=True):
    st.write(f"""
    - **Traditional Cycle Time**: {FIXED_CYCLE_TIME}s per lane
    - **Yellow Transition Time**: {YELLOW_TRANSITION_TIME}s
    - **Emergency Vehicle Priority**: Enabled
    """)
    st.dataframe(df, use_container_width=True)

if st.button("▶️ Start Smart Simulation", type="primary"):
    run_synchronized_simulation(df)