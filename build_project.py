#!/usr/bin/env python3
"""
Build script for Employee Events Dashboard project.
Creates sample data, database, and trained ML model.
"""

import json
import sqlite3
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os

# Create project structure
project_dirs = [
    'assets',
    'employee_events',
    'base_components',
    'combined_components'
]

for dir_name in project_dirs:
    Path(dir_name).mkdir(exist_ok=True)

# Generate sample data
print("Generating sample data...")

# Employee names
employee_names = [
    "John Smith", "Sarah Johnson", "Mike Davis", "Lisa Brown", "Chris Wilson",
    "Amy Taylor", "Robert Miller", "Jennifer Garcia", "David Martinez", "Michelle Rodriguez",
    "Kevin Anderson", "Jessica Thomas", "Brian Jackson", "Ashley White", "Daniel Harris",
    "Stephanie Martin", "Matthew Thompson", "Laura Clark", "Ryan Lewis", "Amanda Lee",
    "James Walker", "Samantha Hall", "Andrew Allen", "Nicole Young", "Joseph King"
]

# Team information
teams = [
    {"name": "Production Team A", "shift": "Morning", "manager": "Alice Johnson"},
    {"name": "Quality Control", "shift": "Afternoon", "manager": "Bob Smith"},
    {"name": "Maintenance", "shift": "Evening", "manager": "Carol Davis"},
    {"name": "Logistics", "shift": "Night", "manager": "David Wilson"},
    {"name": "Engineering", "shift": "Morning", "manager": "Eva Martinez"}
]

# Create database
print("Creating database...")
db_path = Path("employee_events.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create tables
cursor.execute("""
    CREATE TABLE IF NOT EXISTS team (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        shift TEXT NOT NULL,
        manager TEXT NOT NULL
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS employee (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        team_id INTEGER NOT NULL,
        FOREIGN KEY (team_id) REFERENCES team(id)
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS employee_events (
        id INTEGER PRIMARY KEY,
        employee_id INTEGER NOT NULL,
        event_date DATE NOT NULL,
        positive_events INTEGER DEFAULT 0,
        negative_events INTEGER DEFAULT 0,
        FOREIGN KEY (employee_id) REFERENCES employee(id)
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY,
        employee_id INTEGER NOT NULL,
        note TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (employee_id) REFERENCES employee(id)
    )
""")

# Insert team data
for i, team in enumerate(teams, 1):
    cursor.execute(
        "INSERT OR REPLACE INTO team (id, name, shift, manager) VALUES (?, ?, ?, ?)",
        (i, team["name"], team["shift"], team["manager"])
    )

# Insert employee data
for i, name in enumerate(employee_names, 1):
    team_id = ((i - 1) % len(teams)) + 1
    cursor.execute(
        "INSERT OR REPLACE INTO employee (id, name, team_id) VALUES (?, ?, ?)",
        (i, name, team_id)
    )

# Generate performance events data
print("Generating performance events...")
np.random.seed(42)
start_date = datetime.now() - timedelta(days=365)

events_data = []
for employee_id in range(1, len(employee_names) + 1):
    for days_ago in range(365):
        event_date = start_date + timedelta(days=days_ago)
        
        # Generate realistic performance data
        positive_events = np.random.poisson(2)  # Average 2 positive events per day
        negative_events = np.random.poisson(0.5)  # Average 0.5 negative events per day
        
        events_data.append({
            'employee_id': employee_id,
            'event_date': event_date.strftime('%Y-%m-%d'),
            'positive_events': positive_events,
            'negative_events': negative_events
        })

# Insert events data
for event in events_data:
    cursor.execute(
        "INSERT INTO employee_events (employee_id, event_date, positive_events, negative_events) VALUES (?, ?, ?, ?)",
        (event['employee_id'], event['event_date'], event['positive_events'], event['negative_events'])
    )

# Generate notes data
print("Generating notes...")
sample_notes = [
    "Excellent performance on quarterly review",
    "Showed great teamwork during project",
    "Needs improvement in time management",
    "Outstanding problem-solving skills",
    "Great mentor to junior staff",
    "Innovative approach to challenges",
    "Strong communication skills",
    "Reliable and consistent performer",
    "Shows initiative in new projects",
    "Technical expertise highly valued"
]

for employee_id in range(1, len(employee_names) + 1):
    # Add 2-3 notes per employee
    num_notes = np.random.randint(2, 4)
    for _ in range(num_notes):
        note = np.random.choice(sample_notes)
        cursor.execute(
            "INSERT INTO notes (employee_id, note) VALUES (?, ?)",
            (employee_id, note)
        )

conn.commit()

# Create training data for ML model
print("Creating ML model...")
df = pd.read_sql_query("""
    SELECT 
        e.id as employee_id,
        e.name as employee_name,
        SUM(ev.positive_events) as total_positive,
        SUM(ev.negative_events) as total_negative,
        t.name as team_name
    FROM employee e
    JOIN employee_events ev ON e.id = ev.employee_id
    JOIN team t ON e.team_id = t.id
    GROUP BY e.id, e.name, t.name
""", conn)

# Create synthetic recruitment risk labels (for demonstration)
# In a real scenario, this would be historical data
df['recruitment_risk'] = (
    (df['total_negative'] > df['total_positive']) |
    (df['total_positive'] < df['total_positive'].quantile(0.3))
).astype(int)

# Prepare features for ML model
X = df[['total_positive', 'total_negative']]
y = df['recruitment_risk']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Save model
model_path = Path("assets/model.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

# Test model accuracy
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Model training accuracy: {train_score:.2f}")
print(f"Model test accuracy: {test_score:.2f}")

conn.close()

print("Project build completed successfully!")
print(f"Database created: {db_path}")
print(f"Model saved: {model_path}")
print(f"Total employees: {len(employee_names)}")
print(f"Total teams: {len(teams)}")
print(f"Total events: {len(events_data)}")