#!/usr/bin/env python3
"""
Employee Events Dashboard - FastHTML Application
A comprehensive dashboard for monitoring employee performance and recruitment risk.
"""

import sqlite3
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import io
import base64
from typing import Optional, Dict, List, Any

from fasthtml.common import *
from fasthtml import FastHTML

# Initialize FastHTML app
app = FastHTML(
    hdrs=[
        Link(rel="stylesheet", href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css"),
        Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"),
        Script(src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"),
        Script(src="https://unpkg.com/htmx.org@1.9.10")
    ]
)

# Database and model paths
DB_PATH = Path("employee_events.db")
MODEL_PATH = Path("assets/model.pkl")

# Load ML model
def load_model():
    """Load the trained logistic regression model."""
    try:
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Database helper functions
def get_db_connection():
    """Get database connection."""
    return sqlite3.connect(DB_PATH)

def get_employees():
    """Get all employees from database."""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("""
            SELECT e.id, e.name, t.name as team_name, t.shift, t.manager
            FROM employee e
            JOIN team t ON e.team_id = t.id
            ORDER BY e.name
        """, conn)
        return df.to_dict('records')
    finally:
        conn.close()

def get_teams():
    """Get all teams from database."""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM team ORDER BY name", conn)
        return df.to_dict('records')
    finally:
        conn.close()

def get_employee_events(employee_id: int, days: int = 30):
    """Get employee events for the last N days."""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("""
            SELECT 
                event_date,
                positive_events,
                negative_events,
                (positive_events - negative_events) as net_events
            FROM employee_events
            WHERE employee_id = ? 
            AND event_date >= date('now', '-{} days')
            ORDER BY event_date
        """.format(days), conn, params=(employee_id,))
        return df
    finally:
        conn.close()

def get_employee_summary(employee_id: int):
    """Get employee summary statistics."""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("""
            SELECT 
                e.name,
                t.name as team_name,
                t.shift,
                t.manager,
                SUM(ev.positive_events) as total_positive,
                SUM(ev.negative_events) as total_negative,
                AVG(ev.positive_events) as avg_positive,
                AVG(ev.negative_events) as avg_negative
            FROM employee e
            JOIN team t ON e.team_id = t.id
            JOIN employee_events ev ON e.id = ev.employee_id
            WHERE e.id = ?
            GROUP BY e.id, e.name, t.name, t.shift, t.manager
        """, conn, params=(employee_id,))
        return df.iloc[0] if not df.empty else None
    finally:
        conn.close()

def get_employee_notes(employee_id: int):
    """Get employee notes."""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("""
            SELECT note, created_at
            FROM notes
            WHERE employee_id = ?
            ORDER BY created_at DESC
        """, conn, params=(employee_id,))
        return df.to_dict('records')
    finally:
        conn.close()

def predict_recruitment_risk(employee_id: int):
    """Predict recruitment risk for an employee."""
    model = load_model()
    if not model:
        return None
    
    summary = get_employee_summary(employee_id)
    if not summary:
        return None
    
    # Prepare features
    features = np.array([[summary['total_positive'], summary['total_negative']]])
    
    # Get prediction probability
    risk_prob = model.predict_proba(features)[0][1]
    return risk_prob

def create_performance_chart(employee_id: int, days: int = 30):
    """Create a performance chart for an employee."""
    events_df = get_employee_events(employee_id, days)
    
    if events_df.empty:
        return None
    
    # Create matplotlib figure
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert date strings to datetime
    events_df['event_date'] = pd.to_datetime(events_df['event_date'])
    
    # Plot positive and negative events
    ax.plot(events_df['event_date'], events_df['positive_events'], 
            color='#28a745', marker='o', label='Positive Events', linewidth=2)
    ax.plot(events_df['event_date'], events_df['negative_events'], 
            color='#dc3545', marker='o', label='Negative Events', linewidth=2)
    ax.plot(events_df['event_date'], events_df['net_events'], 
            color='#ffc107', marker='s', label='Net Events', linewidth=2)
    
    ax.set_xlabel('Date', color='white')
    ax.set_ylabel('Number of Events', color='white')
    ax.set_title(f'Performance Events - Last {days} Days', color='white', fontsize=16)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set background color
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    
    # Convert to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', facecolor='#1a1a1a')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.read()).decode()
    plt.close()
    
    return img_str

# Route handlers
@app.get("/")
def home():
    """Main dashboard page."""
    employees = get_employees()
    teams = get_teams()
    
    return Html(
        Head(
            Title("Employee Events Dashboard"),
            Meta(charset="utf-8"),
            Meta(name="viewport", content="width=device-width, initial-scale=1")
        ),
        Body(
            # Navigation
            Nav(
                Div(
                    A("Employee Events Dashboard", href="/", 
                      cls="navbar-brand fs-4 fw-bold text-primary"),
                    cls="container"
                ),
                cls="navbar navbar-expand-lg navbar-dark bg-dark mb-4"
            ),
            
            # Main content
            Div(
                # Header
                Div(
                    H1("Employee Performance Dashboard", cls="text-center mb-4"),
                    P("Monitor employee performance and analyze recruitment risk using machine learning.",
                      cls="text-center text-muted mb-4"),
                    cls="row"
                ),
                
                # Controls
                Div(
                    Div(
                        H4("Select Employee", cls="mb-3"),
                        Form(
                            Select(
                                Option("Choose an employee...", value="", selected=True),
                                *[Option(f"{emp['name']} ({emp['team_name']})", value=str(emp['id']))
                                  for emp in employees],
                                name="employee_id",
                                cls="form-select mb-3",
                                hx_post="/employee_dashboard",
                                hx_target="#dashboard-content",
                                hx_trigger="change"
                            ),
                            cls="mb-4"
                        ),
                        cls="col-md-6"
                    ),
                    Div(
                        H4("Team Overview", cls="mb-3"),
                        Div(
                            *[Div(
                                H6(team['name'], cls="card-title"),
                                P(f"Shift: {team['shift']}", cls="card-text small"),
                                P(f"Manager: {team['manager']}", cls="card-text small"),
                                cls="card-body"
                            ) for team in teams[:3]],
                            cls="row"
                        ),
                        cls="col-md-6"
                    ),
                    cls="row mb-4"
                ),
                
                # Dashboard content (will be populated by HTMX)
                Div(
                    Div(
                        I(cls="bi bi-person-plus-fill me-2"),
                        "Select an employee to view their performance dashboard",
                        cls="alert alert-info text-center"
                    ),
                    id="dashboard-content"
                ),
                
                cls="container"
            ),
            
            # Footer
            Footer(
                Div(
                    P("Employee Events Dashboard © 2025", cls="text-center text-muted"),
                    cls="container"
                ),
                cls="bg-dark text-white py-3 mt-5"
            )
        )
    )

@app.post("/employee_dashboard")
def employee_dashboard(employee_id: int):
    """Load employee dashboard content."""
    if not employee_id:
        return Div("Please select an employee", cls="alert alert-warning")
    
    # Get employee data
    summary = get_employee_summary(employee_id)
    if not summary:
        return Div("Employee not found", cls="alert alert-danger")
    
    notes = get_employee_notes(employee_id)
    risk_prob = predict_recruitment_risk(employee_id)
    chart_img = create_performance_chart(employee_id)
    
    # Risk assessment
    risk_level = "Low"
    risk_color = "success"
    if risk_prob:
        if risk_prob > 0.7:
            risk_level = "High"
            risk_color = "danger"
        elif risk_prob > 0.4:
            risk_level = "Medium"
            risk_color = "warning"
    
    return Div(
        # Employee header
        Div(
            Div(
                H3(f"{summary['name']}", cls="mb-1"),
                P(f"Team: {summary['team_name']} | Shift: {summary['shift']} | Manager: {summary['manager']}", 
                  cls="text-muted"),
                cls="col-md-8"
            ),
            Div(
                Div(
                    H5("Recruitment Risk", cls="mb-1"),
                    Span(f"{risk_level}", cls=f"badge bg-{risk_color} fs-6"),
                    P(f"{risk_prob:.1%}" if risk_prob else "N/A", cls="small text-muted"),
                    cls="text-end"
                ),
                cls="col-md-4"
            ),
            cls="row align-items-center mb-4 p-3 bg-dark rounded"
        ),
        
        # Performance metrics
        Div(
            Div(
                Div(
                    I(cls="bi bi-arrow-up-circle-fill text-success me-2"),
                    H4(f"{int(summary['total_positive'])}", cls="mb-1"),
                    P("Total Positive Events", cls="text-muted small"),
                    cls="card-body text-center"
                ),
                cls="card bg-dark"
            ),
            Div(
                Div(
                    I(cls="bi bi-arrow-down-circle-fill text-danger me-2"),
                    H4(f"{int(summary['total_negative'])}", cls="mb-1"),
                    P("Total Negative Events", cls="text-muted small"),
                    cls="card-body text-center"
                ),
                cls="card bg-dark"
            ),
            Div(
                Div(
                    I(cls="bi bi-graph-up text-info me-2"),
                    H4(f"{summary['avg_positive']:.1f}", cls="mb-1"),
                    P("Avg Positive/Day", cls="text-muted small"),
                    cls="card-body text-center"
                ),
                cls="card bg-dark"
            ),
            Div(
                Div(
                    I(cls="bi bi-graph-down text-warning me-2"),
                    H4(f"{summary['avg_negative']:.1f}", cls="mb-1"),
                    P("Avg Negative/Day", cls="text-muted small"),
                    cls="card-body text-center"
                ),
                cls="card bg-dark"
            ),
            cls="row g-3 mb-4"
        ),
        
        # Performance chart
        Div(
            Div(
                H5("Performance Trends", cls="card-title"),
                Img(src=f"data:image/png;base64,{chart_img}", cls="img-fluid") if chart_img else P("No chart data available"),
                cls="card-body"
            ),
            cls="card bg-dark mb-4"
        ),
        
        # Notes section
        Div(
            Div(
                H5("Manager Notes", cls="card-title"),
                Div(
                    *[Div(
                        P(note['note'], cls="mb-1"),
                        Small(f"Added: {note['created_at']}", cls="text-muted"),
                        cls="border-bottom pb-2 mb-2"
                    ) for note in notes[:5]] if notes else [P("No notes available", cls="text-muted")],
                    cls="card-text"
                ),
                cls="card-body"
            ),
            cls="card bg-dark"
        )
    )

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)