import plotly.graph_objects as go
import numpy as np
import pickle
import torch
def export_dual_theme_plotly(data, base_filename="planning_plot"):
    planning_steps = sorted(data.keys())
    algorithm_names = list(data[planning_steps[0]].keys())
    
    # Pre-calculate global Y-limits to prevent axis "jumping"
    all_values = [val for step in data.values() for series in step.values() for val in series]
    y_min, y_max = min(all_values), max(all_values)
    # Add 5% padding
    padding = (y_max - y_min) * 0.05
    y_range = [y_min - padding, y_max + padding]

    for theme in ['plotly_white', 'plotly_dark']:
        frames = []
        for step in planning_steps:
            frame_data = [
                go.Scatter(x=np.arange(len(data[step][name])), 
                           y=data[step][name], 
                           name=name,
                           mode='lines',
                           line=dict(width=2)) 
                for name in algorithm_names
            ]
            frames.append(go.Frame(data=frame_data, name=str(step)))

        fig = go.Figure(data=frames[0].data, frames=frames)

        # Slider config: immediate redraw, no animation
        sliders = [{
            "active": 0,
            "currentvalue": {"prefix": "Planning Steps: "},
            "pad": {"t": 50},
            "steps": [
                {
                    "args": [[f.name], {
                        "frame": {"duration": 0, "redraw": True}, 
                        "mode": "immediate",
                        "fromcurrent": True,
                        "transition": {"duration": 0}
                    }],
                    "label": f.name,
                    "method": "animate"
                } for f in frames
            ],
        }]

        fig.update_layout(
            template=theme,
            sliders=sliders,
            xaxis_title="Time Steps",
            yaxis_title="Cumulative Reward",
            yaxis=dict(range=y_range), # Fixed range for stability
            # margin=dict(l=20, r=20, t=40, b=20),
            # --- TRANSPARENCY SETTINGS ---
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )

        suffix = "light" if theme == "plotly_white" else "dark"
        # div_id is used by the JavaScript arrow-key listener
        fig.write_html(f"{base_filename}_{suffix}.html", 
                       full_html=False, 
                       include_plotlyjs='cdn',
                       div_id=f"plotly-{suffix}")

# export_dual_theme_plotly(data)

export_dual_theme_plotly(data)