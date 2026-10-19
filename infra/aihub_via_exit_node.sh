#!/bin/bash
# Wait for a tailnet exit node, route this box through it, prove AI Hub accepts the IP, then download.
#   bash infra/aihub_via_exit_node.sh <exit-node-name> <datasetkey> <filekey...>
# Exit node = a PC on a Korean ISP line; AI Hub refuses AWS IPs as "overseas" even in ap-northeast-2.
set -euo pipefail
NODE=$1; DS=$2; shift 2
ROOT=${AIHUB_ROOT:-$HOME/aihub}
. "$ROOT/.key.env"
until tailscale exit-node list 2>/dev/null | grep -q -w "$NODE"; do sleep 30; done
echo "exit node $NODE offered $(date -u +%FT%TZ)"
# allow-lan-access keeps the VPC (instance metadata, private subnet) reachable
sudo -n tailscale set --exit-node="$NODE" --exit-node-allow-lan-access=true
sleep 5
curl -s -m 10 https://ipinfo.io/json | grep -E '"(city|country|org)"'
code=$(curl -s -m 20 -r 0-200 -o /dev/null -w '%{http_code}' -H "apikey:$AIHUB_APIKEY" \
  "https://api.aihub.or.kr/down/0.6/$DS.do?fileSn=$1" || true)
echo "aihub probe: $code"
case "$code" in 2*|3*) ;; *) echo "AI Hub still refuses"; exit 1 ;; esac  # 302 = redirect to the file
bash "$ROOT/aihub_download.sh" "$DS" "$@"
