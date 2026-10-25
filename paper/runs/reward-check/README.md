# Plan C stage 0: reward check

498 one-jamo pairs (font composite, same scene, one jamo changed). Pass = the reward ranks the right image
higher in at least 95% of pairs. Images were made once; every version below only rescored them.

| version | what the total uses | right | vowel | tense | initial | final | unseen syllable |
|---|---|---:|---:|---:|---:|---:|---:|
| v1 | 0.6 glyph yes/no + 0.3 dual-OCR word + extra | 0.785 | 0.665 | 0.812 | 0.889 | 0.846 | 0.588 |
| v2 | + whole-crop likelihood, crop h 192 | 0.829 | 0.740 | 0.854 | 0.919 | 0.862 | 0.635 |
| v3 | same, crop h 384 | 0.835 | 0.723 | 0.896 | 0.919 | 0.877 | 0.647 |
| v4 | worst token instead of mean | 0.809 | 0.699 | 0.865 | 0.899 | 0.846 | 0.588 |
| v5 | per-syllable cell likelihood + 2 x whole-crop likelihood | **0.894** | 0.809 | 0.927 | 0.970 | 0.923 | 0.859 |

Still FAIL. The v5 weights were chosen on these same 498 pairs, so 0.894 is optimistic.

What is left: the failures are mostly syllables Qwen3-VL cannot read even when they are right (the right
syllable's log-prob is below -5 in half of them: 퉁, rare and unseen syllables), and vowel pairs
(은/운, 을/울, 기/긔). Cells come from the known placement, so v5 also needs a syllable detector before it can
score model-drawn images.

Files: check.json (v1), check-v5.json (v5 run; its stored total used the earlier v5 weights, 0.869),
scripts/reward.py.
