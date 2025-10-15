#!/usr/bin/env bash
set -o errexit

# Upgrade pip and build tools
pip install --upgrade pip setuptools wheel

# Install project dependencies
pip install -r requirements.txt
