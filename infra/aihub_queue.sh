#!/bin/bash
# Full download queue after the pilot: rest of AI Hub 105, then the useful parts of 81.
# Order: validation first (eval), then one zip of each sign type / book set for variety, then the rest.
# 81: text-in-the-wild scenes and printed glyphs; handwriting and the 140 GB augmentation set are skipped.
#   setsid nohup bash infra/aihub_queue.sh > ~/aihub/queue.log 2>&1 &
# Each fetch reads its script fresh, so deploy a fixed fetcher under a new name while a queue runs.
set -uo pipefail
D=$(dirname "$0")
F=${AIHUB_FETCH:-$HOME/aihub/aihub_fetch.sh}
[ -f "$F" ] || F=$D/aihub_fetch.sh
# wait for an earlier queue (pid in $1) to finish so two downloads never split the home uplink
if [ -n "${1:-}" ]; then while [ -d "/proc/$1" ]; do sleep 60; done; fi
if [ -z "${SKIP_81:-}" ]; then
  bash "$F" 81 67943 67939 67940 67941 67942 255442 255439 255440 255441 || echo "81 queue stopped"
fi
# New scene types first (disk is the limit): every label file, then a varied subset of images.
# 633 packaging: labels + validation + training 1.  ~94 GB
bash "$F" 633 51046 51047 51050 51051 51048 || echo "633 queue stopped"
# 71553 menus: all labels, all validation images, four regions of training.  ~76 GB
bash "$F" 71553 485581 485582 485583 485584 485585 485586 485587 485588 485589 485590 485591 485592 485593 \
  485607 485608 485609 485610 485611 485612 485613 485614 485615 485616 485617 485618 485619 \
  485594 485595 485596 485597 485598 485599 485600 485601 485602 485603 485604 485605 485606 \
  485569 485573 485580 485576 || echo "71553 queue stopped"
# 71725 tourism: all labels, posters (PS) and tables (TB); maps, tickets and leaflets skipped.  ~96 GB
bash "$F" 71725 524748 524749 524750 524751 524752 524753 524754 524755 \
  524764 524765 524766 524767 524768 524769 524770 524771 \
  524758 524760 524761 524762 524763 524744 524745 524746 524747 524740 || echo "71725 queue stopped"
bash "$F" 105 68073 37321 37318 37319 37320 \
  37306 37307 37309 37311 37312 37313 37304 37315 37300 68071 \
  37301 37302 37303 37305 37308 37310 37316 || echo "105 queue stopped"
echo "QUEUE DONE $(date -u +%FT%TZ)"
