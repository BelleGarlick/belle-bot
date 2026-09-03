#!/bin/bash
set -e

cd belle_bot/houston/frontend
npx vite build
echo "Saved to frontend/dist"
