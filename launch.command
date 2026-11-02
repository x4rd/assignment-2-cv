#!/bin/zsh
# Double-click on macOS to start the local model server and open the app.
set -u
cd "${0:A:h}"

fail() {
  print -u2 "$1"
  if [[ -t 0 ]]; then
    read '?Press Enter to close.'
  fi
  exit 1
}

APP_PORT=${PORT:-8000}
APP_URL="http://127.0.0.1:${APP_PORT}"
server_ready() {
  /usr/bin/curl --silent --fail --max-time 2 "${APP_URL}/api/health" | /usr/bin/grep -q '"ready": true'
}

if server_ready; then
  print "Object Lens is already running at ${APP_URL}"
  /usr/bin/open "${APP_URL}" || fail 'Could not open the browser.'
  exit 0
fi

if [[ ! -x .venv/bin/python ]]; then
  command -v python3 >/dev/null 2>&1 || fail 'Python 3 is required. Install it, then try again.'
  print 'Creating the local Python environment...'
  python3 -m venv .venv || fail 'Could not create the Python environment.'
fi
APP_PYTHON=.venv/bin/python

if ! "$APP_PYTHON" -c 'import tensorflow, numpy, PIL' >/dev/null 2>&1; then
  print 'Installing Python packages. The first launch may take a few minutes...'
  "$APP_PYTHON" -m pip install -r requirements.txt || fail 'Package installation failed. Check the error above and try again.'
fi

print "Starting Object Lens at ${APP_URL}"
(
  for attempt in {1..120}; do
    if server_ready; then
      /usr/bin/open "${APP_URL}" || print -u2 'Could not open the browser. Use the URL above.'
      exit 0
    fi
    sleep 1
  done
  print -u2 "The server did not become ready. Check the errors above."
) &
"${APP_PYTHON}" server.py
