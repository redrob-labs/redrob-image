#!/bin/bash
# Download AI Hub files one at a time WITHOUT aihubshell's merge step, which loses data:
# it escapes non-ASCII names with printf %q, writes a 0-byte zip, then deletes the parts.
#   bash infra/aihub_fetch.sh <datasetkey> <filekey...>
# Per filekey: curl the tar (resumable) -> untar -> join .partN by numeric offset -> verify zip
# -> drop parts and tar -> .done/<filekey>. infra/aihub_extract.py then unzips and deletes the zip.
# Needs ~/aihub/.key.env (export AIHUB_APIKEY=..., mode 600) and approval on aihub.or.kr.
set -euo pipefail
DS=$1; shift
ROOT=${AIHUB_ROOT:-$HOME/aihub}
. "$ROOT/.key.env"
mkdir -p "$ROOT/$DS/.done" "$ROOT/$DS/.tar"
cd "$ROOT/$DS"
URL="https://api.aihub.or.kr/down/0.6/$DS.do"
for k in "$@"; do
  [ -f ".done/$k" ] && { echo "skip $k"; continue; }
  echo "== $k $(date -u +%FT%TZ)"
  T=".tar/$k.tar"
  # .complete / .untarred markers let a restart skip the download and the untar
  if [ ! -f ".tar/$k.complete" ]; then
    for try in 1 2 3 4 5; do
      code=$(curl -sS -L -C - -o "$T" -H "apikey:$AIHUB_APIKEY" -w '%{http_code}' "$URL?fileSn=$k" || true)
      case "$code" in 200|206|416) break ;; esac
      echo "try $try: http $code"; sleep 60
    done
    if ! tar --quoting-style=literal -tf "$T" > ".tar/$k.list" 2>/dev/null; then
      echo "FAILED $k: not a tar (http $code): $(head -c 200 "$T")"; rm -f "$T"; exit 1
    fi
    touch ".tar/$k.complete"
  fi
  if [ ! -f ".tar/$k.untarred" ]; then
    tar -xf "$T"
    touch ".tar/$k.untarred"
  fi
  # join every <name>.partN group listed in this tar, by numeric offset
  sed -n 's/\.part[0-9]*$//p' ".tar/$k.list" | sort -u | while IFS= read -r z; do
    ls -1 "$z".part* | awk -F'.part' '{print $NF"\t"$0}' | sort -n | cut -f2- | tr '\n' '\0' | xargs -0 cat > "$z.joining"
    python3 -c "import sys,zipfile; zipfile.ZipFile(sys.argv[1]).close()" "$z.joining"
    mv -f "$z.joining" "$z"
    rm -f "$z".part*
    echo "joined $z $(stat -c %s "$z")"
  done
  rm -f "$T"
  touch ".done/$k"
done
echo "ALL DONE $DS $(date -u +%FT%TZ)"
