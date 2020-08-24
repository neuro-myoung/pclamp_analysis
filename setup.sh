#!/bin/bash

mkdir -p ~/.streamlit/

cat <<\EOT > ~/.streamlit/credentials.toml
[general]
email = "michael.young@duke.edu
[server]
headless = true
enableCORS=false
port = $PORT
EOT
