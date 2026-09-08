#!/bin/bash
set -e

cd houston/frontend
npx vite build
echo "Saved to frontend/dist"
