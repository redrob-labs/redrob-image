#!/usr/bin/env bash
# OFL Korean fonts from google/fonts for the syllable classifier (scripts/syllable_cls.py).
# Noto Sans KR is NOT fetched here on purpose: the stage-0 pairs are drawn in it, so it stays held out.
set -u
DST="${1:-$HOME/.local/share/fonts/ko-train}"
mkdir -p "$DST"
BASE=https://raw.githubusercontent.com/google/fonts/main/ofl
for f in \
  nanumgothic/NanumGothic-Regular.ttf nanumgothic/NanumGothic-Bold.ttf nanumgothic/NanumGothic-ExtraBold.ttf \
  nanummyeongjo/NanumMyeongjo-Regular.ttf nanummyeongjo/NanumMyeongjo-Bold.ttf \
  nanumgothiccoding/NanumGothicCoding-Regular.ttf \
  nanumpenscript/NanumPenScript-Regular.ttf nanumbrushscript/NanumBrushScript-Regular.ttf \
  gothica1/GothicA1-Light.ttf gothica1/GothicA1-Regular.ttf gothica1/GothicA1-Bold.ttf gothica1/GothicA1-Black.ttf \
  gowundodum/GowunDodum-Regular.ttf gowunbatang/GowunBatang-Regular.ttf gowunbatang/GowunBatang-Bold.ttf \
  ibmplexsanskr/IBMPlexSansKR-Regular.ttf ibmplexsanskr/IBMPlexSansKR-Bold.ttf \
  blackhansans/BlackHanSans-Regular.ttf dohyeon/DoHyeon-Regular.ttf jua/Jua-Regular.ttf \
  gaegu/Gaegu-Regular.ttf gaegu/Gaegu-Bold.ttf sunflower/Sunflower-Medium.ttf songmyung/SongMyung-Regular.ttf \
  dokdo/Dokdo-Regular.ttf gugi/Gugi-Regular.ttf yeonsung/YeonSung-Regular.ttf poorstory/PoorStory-Regular.ttf \
  stylish/Stylish-Regular.ttf himelody/HiMelody-Regular.ttf cutefont/CuteFont-Regular.ttf \
  gamjaflower/GamjaFlower-Regular.ttf singleday/SingleDay-Regular.ttf kiranghaerang/KirangHaerang-Regular.ttf \
  eastseadokdo/EastSeaDokdo-Regular.ttf blackandwhitepicture/BlackAndWhitePicture-Regular.ttf ; do
  out="$DST/$(basename "$f" | tr -d '[]')"
  [ -s "$out" ] && continue
  curl -fsSL --retry 3 -o "$out" "$BASE/$f" || { echo "miss $f"; rm -f "$out"; }
done
ls "$DST" | wc -l
