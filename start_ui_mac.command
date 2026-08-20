#!/bin/bash
cd -- "$(dirname -- "$0")"

bash start_ui.sh
status=$?
if [[ $status -ne 0 ]]; then
  echo
  echo "FLUID-Space stopped with exit code $status."
  read -r -p "Press Return to close..."
fi
exit $status
