# Houston Overview

Houston is the management and visualization component for `belle-bot`. It consists of a backend server, a frontend dashboard, and a client library.

## Components

### 1. Server (`/houston/server`)
A FastAPI-based backend that handles:
- API requests for replays and other bot data.
- Persistence of events and logs.
- Core logic for replaying recorded data.

### 2. Frontend (`/houston/frontend`)
A modern React application built with TypeScript and Vite. It provides:
- A user interface for exploring and managing replays.
- Visualization tools for bot activities.

### 3. Client (`/houston/client`)
A Python client library that allows other parts of `belle-bot` to communicate with the Houston API.

### 4. Scripts (`/houston/scripts`)
Utility scripts for building and running the different parts of Houston.

## Project Structure

```text
houston/
├── client/     # Python client library
├── docs/       # Documentation (you are here)
├── frontend/   # React + Vite frontend
├── scripts/    # Build and run scripts
├── server/     # FastAPI backend
└── readme.md   # Quick start guide
```
