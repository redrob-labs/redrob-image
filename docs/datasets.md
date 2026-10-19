# Dataset sources for Korean text rendering and realism

Checked 2026-10-19. Every link was fetched when this file was written (AI Hub pages need a
browser User-Agent; FAQ page 2 is reached through the list's pager). Sizes for AI Hub are the
page's own `s3FileSize` value. This is research, not legal advice: the human owner confirms terms
on the application screen before downloading.

## (a) AI Hub (aihub.or.kr)

### Terms that apply to every NIA-built ("지능정보산업 인프라 조성") dataset below

Source: [AI Hub 데이터 이용정책](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105)
and [AI Hub 자주찾는질문](https://www.aihub.or.kr/aihubnews/faq/list.do?currMenu=146&topMenu=104).

| Question | Answer | Quote |
|---|---|---|
| Commercial use? | Yes for R&D and training | Policy: "지능형 제품・서비스, 챗봇 등 다양한 분야에서 영리적・비영리적 연구・개발 목적으로 활용할 수 있습니다." / "본 AI데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다." |
| Distribute / sell the trained model? | Yes, with attribution | FAQ "AI 허브 데이터를 영리적(상업적) 목적으로 사용할 수 있나요?": "AI 허브 데이터를 학습에 활용하여 개발한 AI 모델, 서비스 및 연구 결과물 등 2차 저작물은 영리·비영리 목적으로 자유롭게 활용하거나 판매·배포할 수 있습니다. 이 경우 활용한 데이터셋의 정식 명칭과 AI 허브(aihub.or.kr)를 출처로 표기해 주시기 바랍니다." |
| Redistribute data? | No, not even edited copies | FAQ: "원본 데이터 자체를 제3자에게 제공하거나 배포하는 행위는 허용되지 않습니다." Policy item 5: no "열람하게 하거나 제공, 양도, 대여, 판매". Selling the dataset needs a separate agreement with the builder. |
| Attribution in derivatives | Required | Policy item 1: "반드시 한국지능정보사회진흥원의 사업결과임을 밝혀야 하며, 본 AI데이터 등을 이용한 2차적 저작물에도 동일하게 밝혀야 합니다." |
| Attribution text (FAQ, papers) | Must contain dataset name + "AI 허브(aihub.or.kr)" | KO: "이 연구는 과학기술정보통신부의 재원으로 한국지능정보사회진흥원의 지원을 받아 구축된 "[데이터명]"을 활용하여 수행된 연구입니다. 본 연구에 활용된 데이터는 AI 허브(aihub.or.kr)에서 다운로드할 수 있습니다." EN: "This research (paper) used datasets from The Open AI Dataset Project (AI-Hub, S. Korea). All data information can be accessed through 'AI-Hub (www.aihub.or.kr)'." Minimal original-image examples for reproducibility are allowed in papers. |
| Who may apply | Korean nationals residing in Korea | Every page below: "※ 내국인만 데이터 신청이 가능합니다." FAQ "해외에서도 AI 허브 데이터를 이용할 수 있나요?" (page 2): "대한민국 국적을 가진 국내 거주자(내국인)가 ... 승인을 받은 경우에 한하여 국내에서 이용할 수 있습니다 ... 외국인(국내 체류비자 소지자 포함) 및 재외국민에게는 ... 제공하기 어렵습니다." Policy item 2: foreign-located corporations/individuals need a separate agreement. |
| Overseas transfer / region | Forbidden without agreement; overseas cloud counts as transfer | FAQ: "데이터를 해외에 소재한 서버나 클라우드 환경으로 전송·저장하거나 해당 환경에서 학습에 활용하는 경우에도 국외 반출에 해당하므로, AI 허브 데이터는 국내 서버 및 국내 클라우드 환경에서 이용해 주시기 바랍니다." Policy item 3: "국외 반출을 위해서는 ... 별도로 합의가 필요합니다." |
| Approval period | 3 months, renewable | FAQ: "AI 허브 데이터 이용 승인은 승인일로부터 3개월간 유효합니다." |
| Personal data | Report and delete if found; no re-identification | Policy items 8-9 |

Consequences for us: the applicant must be a Korean national resident in Korea; data, latents
and text crops stay in ap-northeast-2 (the plan's rule is correct and not optional); training on
it must run in Korea (p5 in Seoul, not us-east). Whether a Korea-region AWS account owned by a
foreign-registered entity counts as "국외 소재 법인" is not answered by the FAQ: ask AI Hub
(문의하기) before Path B. Precomputed VAE latents and text embeddings of AI Hub images are
"processed data" and fall under the same no-redistribution and no-transfer rule.

### Datasets

All are standard NIA datasets (policy above), "내국인만" on the page, JSON labels.

| dataSetSn | Name | Size (listed) | Content | Annotation | Fit |
|---|---|---|---|---|---|
| [105](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=105) | 야외 실제 촬영 한글 이미지 (2020, v1.1 2023-02) | 490.3 GB | 450k signboard photos (outdoor 315k: horizontal, vertical, protruding, pole, banner, window; indoor 135k) + 50k book covers (KDC classes), 1600x1200 JPEG | Per-image JSON, word-level bbox + Hangul text | Best real scene-text source |
| [81](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=81) | 한국어 글자체 이미지 (2019, v1.5 2023-07) | 225.7 GB | 11,172 modern syllables in 50 fonts, printed ~2M chars, handwritten ~2.5M chars (by gender/age), words (NIKL 6k frequent words) and news sentences; plus 100k text-in-the-wild photos (signs 17k, trademarks 37k, signboards 30k, other 16k) | COCO-like JSON (info, images, annotations with text, licenses) | Syllable coverage reference and real wild photos; font-rendered part is redundant with our own synth |
| [91](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=91) | 다양한 형태의 한글 문자 OCR (2020) | 42.5 GB | 75k printed forms (50 form types, 10 free fonts, 2,350 syllables), 1.1M handwritten chars/words | JSON | Low priority (documents/handwriting) |
| [63](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=63) | 상표 이미지 및 텍스트 (2020) | 68.2 GB | 1.18M trademark images from KIPRIS (Vienna-code crops) + rejection texts | JSON | Low: logos, little Hangul text labelling; trademark rights in the images themselves |
| [633](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=633) | 의약품, 화장품 패키징 OCR 데이터 (2021) | 162.9 GB | 50k photos of drug/cosmetic packaging | JSON, 4-point polygon + bbox | Good: product labels, dense small Hangul |
| [71553](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71553) | 관광 음식메뉴판 데이터 (2022) | 275.1 GB | 100,113 restaurant menu photos nationwide + menu-name knowledge and EN/JA/ZH translations | JSON, bbox + translation | Good: menus |
| [71725](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71725) | 관광분야 이미지-텍스트 쌍 데이터 (2023) | 770.8 GB | 280k tourism images: fare/time tables, maps 94k, direction signs 49k, tickets 52k, posters/banners 28k | JSON bbox + descriptive text | Good: signage, posters, layout |
| [71730](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71730) | 다중언어 OCR 데이터 (2023) | 496.6 GB | 400,864 images with Korean + EN/ZH/JA; life/culture/public photos and ~27% video subtitle stills (provider-owned video) | Polygon JSON, HD 1280x720 | Medium: bilingual signs; drop the video-still category |
| [605](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=605) | 대용량 손글씨 OCR 데이터 (2021) | 248.5 GB | 106,519 handwritten pages (40+ words each), paper 82% / tablet 18% | JSON bbox | Only if handwriting style is wanted |
| [88](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=88) | 공공행정문서 OCR (2020) | 413.3 GB | 900k scanned Gimhae city documents, 25M words | JSON | Low (old scans) |
| [71299](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71299) | OCR 데이터(공공) (2022) | 162.1 GB | 615k scanned public documents (Changwon, Gimhae, diplomatic archives), 1970s-2010s | Segmentation JSON | Low |
| [71300](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71300) | OCR 데이터(교육) (2022) | 137.1 GB | 128,771 scanned K-12 student handwriting pages | Segmentation JSON | Low; minors' handwriting, avoid |
| [71301](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71301) | OCR 데이터(금융 및 물류) (2022) | 176.5 GB | 50k bank/insurance forms with handwriting + 108k bills of lading | 4-point polygon JSON | Low |
| [632](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=632) | 금융업 특화 문서 OCR 데이터 (2021) | 161.4 GB | 50k bank/insurance forms, printed + handwritten | Polygon + bbox JSON | Low |
| [71969](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71969) | K-스톡 콘텐츠 데이터 (2025) | 52.1 GB | 52,463 SBS broadcast stills (1280x720) with Korean CoT captions, 7 themes (hanok, food, culture...) | JSON captions | Realism/Korean-culture captions only; low resolution, broadcast frames |

Not found as separate datasets: street-sign OCR beyond 105/81/71725 (the 신호등/도로표지판
driving sets 187/188 are object detection without text transcription), Korean product shelf
OCR, poster-only sets.

## (b) Other Korean text-in-image datasets

| Dataset | License | Size | Verdict |
|---|---|---|---|
| [nvidia/OCR-Synthetic-Multilingual-v1](https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v1) (see c) | CC BY 4.0 | ko: 2.27M | Usable |
| [naver-clova-ix/synthdog-ko](https://huggingface.co/datasets/naver-clova-ix/synthdog-ko) | No license on the dataset card; generator code is MIT ([donut](https://github.com/clovaai/donut)). Text from Korean Wikipedia, backgrounds from the repo's `resources/background` (the Donut paper uses ImageNet) | 0.5M | Unclear: do not train on the released images; reuse the MIT generator with our own backgrounds and fonts |
| [WueNLP/Synthdog-Multilingual-100](https://huggingface.co/datasets/WueNLP/Synthdog-Multilingual-100) | Apache-2.0 (card), same SynthDoG backgrounds caveat | ~10k per non-Latin language | Small for Korean; low priority |
| [Jiwon-Kang/OCR-Synthetic-Rendered-Korean-200K](https://huggingface.co/datasets/Jiwon-Kang/OCR-Synthetic-Rendered-Korean-200K) | None stated; derived from other `source_dataset`s | 200k | Unclear, skip |
| [Nexdata-kr/104320-Images-Korean-and-Hindi-OCR-Data-in-Natural-Scenes](https://huggingface.co/datasets/Nexdata-kr/104320-Images-Korean-and-Hindi-OCR-Data-in-Natural-Scenes) | Commercial purchase (sample only on HF) | 76,861 Korean scene images, 556k boxes | Option if budget allows; buy with model-training rights in the contract |
| [hyeongyeolryu/korean_outdoor_ocr](https://huggingface.co/datasets/hyeongyeolryu/korean_outdoor_ocr) | None; folder names (`원천데이터_230216_add`) match AI Hub 105 v1.1 | - | Avoid: unauthorized AI Hub redistribution |
| [jeina/korean_ocr_public](https://huggingface.co/datasets/jeina/korean_ocr_public) | None; card says "공공행정문서" (AI Hub 88) | - | Avoid: same reason |
| [AbdullahRian/Korean.OCR.Img.text.pair](https://huggingface.co/datasets/AbdullahRian/Korean.OCR.Img.text.pair) | None, no card | - | Unclear provenance, skip |

ICDAR MLT 2017/2019 (contains Korean) could not be checked: the RRC site failed TLS from this
host. Mark unverified; research-only terms are likely and it is small.

## (c) NVIDIA OCR synthetic data

[nvidia/OCR-Synthetic-Multilingual-v1](https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v1):
the training data of [Nemotron OCR v2](https://huggingface.co/nvidia/nemotron-ocr-v2). 12.26M
images, 5.45 TB total; Korean `ko` 2,269,540 (train 1,814,994 in 78 HDF5 files). Modified
SynthDoG document pages, JPEG in HDF5 with word/line/paragraph quads, text and reading order.

Verdict: commercial training OK. Card: "This dataset is ready for commercial/non-commercial use"
and "Use of the dataset is governed by the Creative Commons Attribution 4.0 International License
(CC BY 4.0)." Attribute NVIDIA (cite `chesler2026ocr_synthetic_multilingual`) in NOTICE/model card.
Caveats: document-page style (not scenes), collection method listed as "Hybrid: Human, Automated,
Synthetic" without naming background sources; sample a shard before planning on it.

Bonus for T3: Nemotron OCR v2 itself supports Korean and is under the NVIDIA Open Model License
(commercially usable), a candidate second OCR engine.

## (d) General / realism photo datasets

| Dataset | License | Size | Captions | Caveats |
|---|---|---|---|---|
| [Spawning/PD12M](https://huggingface.co/datasets/Spawning/PD12M) ([paper](https://arxiv.org/abs/2410.23144)) | Metadata CDLA-Permissive-2.0; images public domain / CC0 | 12.4M | Yes (automated recaptioning) | Images self-hosted by Spawning (no link rot); mixed photos and artworks; English captions only |
| [Spawning/pd-extended](https://huggingface.co/datasets/Spawning/pd-extended) | CDLA-Permissive-2.0 | 10M-100M (HF size tag) | Yes | Same publisher and governance as PD12M; check overlap before mixing |
| [madebyollin/megalith-10m](https://huggingface.co/datasets/madebyollin/megalith-10m) | Link list MIT; images Flickr "no known restrictions", US Gov, CC0, PD Mark | ~10M photos | Via [drawthingsai/megalith-10m-sharecap](https://huggingface.co/datasets/drawthingsai/megalith-10m-sharecap) (CC0, ShareCaptioner) | Links only (download with img2dataset, link rot); Flickr license tags are uploader-asserted; author asks for your own copyright check |
| [common-canvas/commoncatalog-cc-by](https://huggingface.co/datasets/common-canvas/commoncatalog-cc-by) ([paper](https://arxiv.org/abs/2310.16825)) | CC BY 4.0 (per-image license name/url kept) | 10M+ | BLIP-2 synthetic captions | Attribution per image required; YFCC 2014 photos up to 4k; keep only the CC-BY repo (NC/ND variants exist separately) |
| [Open Images V7](https://storage.googleapis.com/openimages/web/factsfigures_v7.html) | Annotations CC BY 4.0; images "listed as having a CC BY 2.0 license" with no warranty | ~9M | Localized Narratives (subset) | Attribution; Google makes no representation on image licenses |
| LAION-5B and derivatives | - | - | - | Avoid (unknown copyright, CSAM findings) |

Recommendation: PD12M first (clean license, hosted images, captions), Megalith-10m second for
pure photographs. Recaption both with our own captioner (T5) so caption style matches the
Korean data.

## (e) Korean text corpora for synthetic rendering

| Corpus | License | Strings-only use |
|---|---|---|
| Korean Wikipedia, [wikimedia/wikipedia `20231101.ko`](https://huggingface.co/datasets/wikimedia/wikipedia) | CC BY-SA 3.0 / GFDL | Fine for sampling words and short phrases; we do not redistribute the rendered images. Attribute Wikipedia in NOTICE |
| [HuggingFaceFW/fineweb-2 `kor_Hang`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | ODC-By 1.0 (on the compilation; underlying web text keeps its own copyright) | Fine for short strings (signboard-length words, names, phrases); 48.6B words / 60.9M docs. Filter PII (phone numbers, names) before rendering |
| NIKL 모두의 말뭉치 ([FAQ](https://kli.korean.go.kr/boards/faqList.do)) | Use agreement, approved purpose only | Not recommended: "말뭉치를 활용한 결과물을 외부에 공개하고자 할 때에는 그 결과물에 말뭉치와 ... 원문 일부 또는 전부가 포함되어서는 안 되며, 공개 전 국립국어원의 사전 승인을 받아야 합니다." Sejong corpus is research-only. An open model that can render corpus strings would need prior NIKL approval |
| Syllable tables (KS X 1001 2,350; Unicode 11,172) | Not copyrightable (character sets) | Generate programmatically |

## (f) Recommendation

| Order | What | From | Human must do |
|---|---|---|---|
| 1 | AI Hub 105 (signboards + book covers) | [dataSetSn=105](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=105) | Korean-national resident applies (phone identity check), accepts policy, downloads straight into a Seoul-region private bucket/instance; renew every 3 months |
| 2 | AI Hub 633, 71553, 71725 (packaging, menus, signs/posters) | links in (a) | Same application; ask AI Hub whether a foreign-registered company's Korea-region cloud account is allowed |
| 3 | NVIDIA OCR-Synthetic-Multilingual-v1 `ko` (start with 2-3 shards) | [HF](https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v1) | Nothing beyond CC BY attribution in NOTICE |
| 4 | PD12M (realism + regularization) | [HF](https://huggingface.co/datasets/Spawning/PD12M) | Nothing; keep the dataset ID in NOTICE |
| 5 | Strings for T6: Wikipedia ko + FineWeb-2 kor_Hang + syllable tables | links in (e) | Confirm CC BY-SA attribution wording in NOTICE |
| 6 | AI Hub 81 (100k wild photos + syllable reference) | [dataSetSn=81](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=81) | Same application |
| Skip | HF re-uploads of AI Hub data, synthdog-ko images, NIKL corpora, LAION | - | - |

Model card / NOTICE must list every AI Hub dataset by its official Korean name plus
"AI 허브(aihub.or.kr)" and the NIA sentence above.
