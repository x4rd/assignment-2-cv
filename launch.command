#!/bin/zsh
# Double-click on macOS to start the local model server and open the app.
set -e
cd "${0:A:h}"

if [[ -x .venv/bin/python ]]; then
  APP_PYTHON=.venv/bin/python
elif [[ -x 'assgnment 1 cv/.venv/bin/python' ]]; then
  APP_PYTHON='assgnment 1 cv/.venv/bin/python'
elif [[ -x '../assgnment 1 cv/.venv/bin/python' ]]; then
  APP_PYTHON='../assgnment 1 cv/.venv/bin/python'
else
  print 'Python environment not found. Follow the fresh-checkout steps in README.md.'
  read '?Press Enter to close.'
  exit 1
fi

APP_PORT=${PORT:-8000}
APP_URL="http://127.0.0.1:${APP_PORT}"
print "Starting Object Lens at ${APP_URL}"
(
  for attempt in {1..45}; do
    if /usr/bin/curl --silent --fail "${APP_URL}/api/health" | /usr/bin/grep -q '"ready": true'; then
      /usr/bin/open "${APP_URL}"
      exit 0
    fi
    sleep 1
  done
) &
"${APP_PYTHON}" server.py
