PyDash Analytics Dashboard

This is the frontend component of the PyDash Augmented Analytics platform, designed as a single-page application (SPA) using HTML, Tailwind CSS, JavaScript, and Plotly.js for interactive data visualization. It communicates with a separate Python-based backend (assumed to be FastAPI) for data processing, chart generation, and AI-driven insights (via Groq/LLMs).

 Key Features

User Authentication: Simple login mechanism.

Data Upload: Upload and process CSV or JSON datasets.

Analysis Builder: Interactive controls for selecting chart type, axes, and aggregation methods.

AI Analyst: Integrates with a Groq API on the backend for quick data insights and feedback.

Interactive Dashboard: Drag-and-drop layout for visualization cards using SortableJS.

PDF Export: Browser-based export functionality for the dashboard layout.

🛠️ Technology Stack

Component

Technology

Description

Frontend

HTML, JavaScript (ES6)

Single file application logic.

Styling

Tailwind CSS (CDN)

Utility-first styling for responsiveness and aesthetics.

Visualization

Plotly.js (CDN)

High-level library for generating interactive charts.

Layout

SortableJS (CDN)

Used for drag-and-drop functionality on the dashboard.

Backend (Assumed)

Python / FastAPI

Handles data persistence, processing (Pandas), chart generation, and AI calls.

📦 Local Setup and Running the App

This project consists of two main parts: the Frontend (this file) and the assumed Backend (FastAPI).

1. Backend Setup (Assumed)

You must ensure your backend service is running and accessible at http://localhost:8000 (as defined in index.html):

Clone the Repository:

git clone <your-repo-url>
cd pydash-analytics


Set up Python Environment (if needed):

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt # Requires FastAPI, Uvicorn, Pandas, Plotly, Groq, etc.


Run the Backend Service:

uvicorn server:app --reload --port 8000


Ensure your backend implements the required routes: /login, /upload, /create_chart, and /get_ai_feedback.

2. Frontend Access

The frontend is a single HTML file and does not require a complex build process.

Open Directly: You can often open index.html directly in your browser.

Use a Simple Server: For reliable cross-origin requests (CORS), it's best to serve the file:

# Requires Python
python -m http.server 3000


Then, navigate to http://localhost:3000/index.html in your browser.

🐳 Docker Deployment

See the included Dockerfile for containerization instructions. This is primarily for the assumed FastAPI backend service.
