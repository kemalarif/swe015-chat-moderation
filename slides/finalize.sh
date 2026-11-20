#!/bin/zsh
# Build the deck (optionally with the demo-video link on slide 1) and export it to PDF with Keynote.
#   ./slides/finalize.sh "https://youtu.be/XXXXXXXX"
# Output: slides/SWE015_ChatModeration.pptx and slides/SWE015_ChatModeration.pdf
set -e
cd "$(dirname "$0")"
VIDEO_URL="${1:-}" node build.js
PPTX="$PWD/SWE015_ChatModeration.pptx"
PDF="$PWD/SWE015_ChatModeration.pdf"
rm -f "$PDF"
osascript - "$PPTX" "$PDF" <<'EOF'
on run argv
  set inPath to POSIX file (item 1 of argv)
  set outPath to POSIX file (item 2 of argv)
  tell application "Keynote"
    set d to open inPath
    delay 2
    export d to outPath as PDF
    close d saving no
  end tell
end run
EOF
echo "PDF: $PDF"
[ -z "$1" ] && echo "NOTE: no video link given – slide 1 still shows the placeholder."
exit 0
