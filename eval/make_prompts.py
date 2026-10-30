#!/usr/bin/env python3
"""Write the text-rendering eval prompt sets (deterministic).

Outputs:
  eval/prompts.jsonl                  200 prompts: 100 unseen (one per unseen_words.txt line) + 100 seen
  eval/prompts_unseen_syllables.jsonl 20 prompts whose targets use syllables from unseen_syllables.txt
  eval/general_prompts.jsonl          50 photo prompts without text (quality regression)

Buckets count Hangul syllables only: 1-2, 3-5, 6-10, 11+.
Scenes are balanced: within every (split, bucket) group of 25, scenes cycle 5 x 5.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 20261019
SCENES = ["signboard", "poster", "book_cover", "product_label", "menu"]
BUCKETS = ["1-2", "3-5", "6-10", "11+"]

# Common signage / label text, very likely present in web-scale training data.
SEEN = {
    "1-2": "약국 커피 할인 세일 병원 은행 식당 출구 입구 빵집 꽃집 우유 라면 김밥 치킨 서점 안경 문구 세탁 휴무 금연 신발 사과 녹차 오픈".split(),
    "3-5": "편의점 주차장 출입금지 화장실 할매국밥 미용실 비상구 세탁소 떡볶이 아메리카노 부동산 노래방 정육점 반찬가게 분식집 문구점 안경원 치킨집 동물병원 영업중 대박할인 신상품 무료배송 유기농 순두부찌개".split(),
    "6-10": [
        "신선한 과일 채소", "전품목 반값 할인", "오늘의 추천 메뉴", "주차 금지 구역입니다", "행복한 하루 되세요",
        "맛있는 우리 동네 빵집", "영업시간 안내", "신규 회원 모집", "할매 순대국밥", "따뜻한 아메리카노",
        "시원한 냉면 개시", "고객 감사 대잔치", "건강한 유기농 식품", "화장실 위치 안내", "금연 구역입니다",
        "엘리베이터 점검 중", "휴대폰 무료 수리", "매주 월요일 휴무", "여름 신상품 입고", "정성 가득 손만두",
        "우리 동네 꽃집", "엄마손 김치찌개", "깨끗한 세탁 서비스", "커피 한 잔의 여유", "새해 복 많이 받으세요",
    ],
    "11+": [
        "이곳은 금연 구역입니다 협조 부탁드립니다", "오늘도 찾아주셔서 감사합니다", "전 품목 최대 오십 퍼센트 할인 행사",
        "주차장 이용 시간은 오후 열 시까지입니다", "매장 내 음식물 반입을 금지합니다", "신선한 재료로 정성껏 만들었습니다",
        "화장실은 복도 끝에 있습니다", "영업시간 오전 아홉 시부터 오후 여섯 시", "안전을 위해 손잡이를 잡아 주세요",
        "따뜻한 커피와 함께 좋은 하루 보내세요", "엘리베이터 점검으로 운행을 중지합니다", "반려동물 동반 출입을 삼가 주세요",
        "새해 복 많이 받으시고 건강하세요", "지금 가입하면 첫 달 무료입니다", "우리 동네 가장 맛있는 김밥집",
        "비상시 이 문을 열고 대피하세요", "손을 깨끗이 씻고 입장해 주세요", "무료 와이파이를 이용하실 수 있습니다",
        "할인 쿠폰은 계산대에서 받으세요", "임대 문의는 아래 번호로 연락 주세요", "공사 중이오니 안전에 유의하시기 바랍니다",
        "맛있는 치킨 배달 주문 받습니다", "금일 휴무 내일 다시 찾아 주세요", "오늘의 추천 메뉴는 된장찌개입니다",
        "정기 세일 기간 동안 전 상품 할인",
    ],
}

# (background ko, text color ko, background en, text color en)
COLORS = [
    ("파란", "흰색", "blue", "white"),
    ("빨간", "노란색", "red", "yellow"),
    ("검은", "흰색", "black", "white"),
    ("초록색", "흰색", "green", "white"),
    ("흰", "검은색", "white", "black"),
    ("노란", "검은색", "yellow", "black"),
    ("남색", "금색", "navy", "gold"),
]


def is_syl(ch: str) -> bool:
    return 0xAC00 <= ord(ch) <= 0xD7A3


def n_syl(s: str) -> int:
    return sum(is_syl(c) for c in s)


def bucket_of(s: str) -> str:
    n = n_syl(s)
    return "1-2" if n <= 2 else "3-5" if n <= 5 else "6-10" if n <= 10 else "11+"


def has_batchim(s: str) -> bool:
    last = [c for c in s if is_syl(c)][-1]
    return (ord(last) - 0xAC00) % 28 != 0


def quote_josa(t: str) -> str:
    """'"할매국밥"이라고' / '"커피"라고'."""
    return f'"{t}"' + ("이라고" if has_batchim(t) else "라고")


def render(scene: str, t: str, c: tuple[str, str, str, str]) -> tuple[str, str]:
    bg, fg, bg_en, fg_en = c
    q = quote_josa(t)
    ko = {
        "signboard": f"{bg} 배경의 가게 간판, {fg} 굵은 글씨로 {q} 적혀 있음, 낮의 거리 사진",
        "poster": f"벽에 붙은 {bg} 포스터, 가운데에 {fg} 큰 글씨로 {q} 적혀 있음",
        "book_cover": f"책상 위에 놓인 {bg} 표지의 책, 표지에 {fg} 제목 글씨로 {q} 적혀 있음",
        "product_label": f"{bg} 병에 붙은 상품 라벨, {fg} 글씨로 {q} 인쇄되어 있음, 제품 사진",
        "menu": f"식당 벽에 걸린 {bg} 메뉴판, {fg} 글씨로 {q} 크게 적혀 있음",
    }[scene]
    en = {
        "signboard": f'A shop signboard with a {bg_en} background, the text "{t}" written in bold {fg_en} letters, daytime street photo',
        "poster": f'A {bg_en} poster on a wall, the text "{t}" written in large {fg_en} letters in the center',
        "book_cover": f'A book with a {bg_en} cover lying on a desk, the title "{t}" written on the cover in {fg_en} letters',
        "product_label": f'A product label on a {bg_en} bottle, the text "{t}" printed in {fg_en} letters, product photo',
        "menu": f'A {bg_en} menu board hanging on a restaurant wall, the text "{t}" written large in {fg_en} letters',
    }[scene]
    return ko, en


def read_list(p: Path) -> list[tuple[str, str]]:
    """(bucket header, entry) pairs from a held-out list."""
    out, head = [], None
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            h = line.lstrip("# ").split(" ")[0]
            if h in BUCKETS:
                head = h
            continue
        out.append((head, line))
    return out


def main() -> None:
    rng = random.Random(SEED)
    unseen_words = read_list(ROOT / "unseen_words.txt")
    unseen_syl = [s for _, s in read_list(ROOT / "unseen_syllables.txt")]
    useto = set(unseen_syl)
    assert len(unseen_words) == 100 and len(unseen_syl) == 120

    def flag(t: str) -> bool:
        return any(c in useto for c in t)

    rows = []
    for split, groups in (
        ("unseen", {b: [w for h, w in unseen_words if h == b] for b in BUCKETS}),
        ("seen", SEEN),
    ):
        for b in BUCKETS:
            words = groups[b]
            assert len(words) == 25, (split, b, len(words))
            for i, t in enumerate(words):
                assert bucket_of(t) == b, (split, b, t)
                if split == "seen":
                    assert not any(u in t for _, u in unseen_words), t
                scene = SCENES[i % 5]
                ko, en = render(scene, t, COLORS[rng.randrange(len(COLORS))])
                rows.append(dict(id=f"{split}-{b}-{i:02d}", split=split, bucket=b, scene=scene,
                                 target_text=t, has_unseen_syllable=flag(t), prompt_ko=ko, prompt_en=en))
    with (ROOT / "prompts.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Unseen-syllable words: 10 = one unseen syllable + common suffix, 10 = 2-3 unseen syllables only.
    suffixes = ["집", "마을", "상회", "식당", "공방", "서점", "카페", "학교", "시장", "약국"]
    picks = rng.sample(unseen_syl, 10 + 25)
    targets = [picks[i] + suffixes[i] for i in range(10)]
    rest = picks[10:]
    k = 0
    for i in range(10):
        n = 2 if i < 6 else 3
        targets.append("".join(rest[k:k + n]))
        k += n
    srows = []
    for i, t in enumerate(targets):
        assert flag(t)
        scene = SCENES[i % 5]
        ko, en = render(scene, t, COLORS[rng.randrange(len(COLORS))])
        srows.append(dict(id=f"usyl-{i:02d}", split="unseen_syllable", bucket=bucket_of(t), scene=scene,
                          target_text=t, has_unseen_syllable=True, prompt_ko=ko, prompt_en=en))
    with (ROOT / "prompts_unseen_syllables.jsonl").open("w", encoding="utf-8") as f:
        for r in srows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    general = [
        "A golden retriever running across a grassy field at sunset, shallow depth of field",
        "Close-up portrait of an elderly woman with silver hair, soft window light",
        "A bowl of ramen with a soft-boiled egg and green onions, top-down food photo",
        "Misty mountain range at dawn with layers of blue ridges",
        "A red vintage bicycle leaning against a white brick wall",
        "A busy night market with string lights, motion-blurred crowd, no signs or text",
        "A tabby cat sleeping on a wooden windowsill in the morning sun",
        "Aerial view of a turquoise coastline with a small sandy beach",
        "A young man in a navy suit walking through a glass office lobby",
        "A ceramic teapot and two cups on a linen tablecloth, still life photo",
        "Autumn forest path covered in orange and yellow leaves",
        "A child blowing soap bubbles in a park, candid photo",
        "Snow-covered traditional Korean hanok roof, winter morning",
        "Macro photo of a dewdrop on a green leaf",
        "A chef plating a dish in a professional kitchen, warm lighting",
        "Desert dunes with long shadows under a clear sky",
        "A woman reading in a cozy armchair next to a fireplace",
        "City skyline at blue hour reflected in a calm river",
        "A plate of colorful macarons arranged in rows",
        "A horse grazing in a foggy meadow",
        "Rain-soaked street with reflections of car headlights, no signage",
        "A smiling grandfather holding a baby, indoor family photo",
        "Fresh vegetables at a farmers market stall, no labels",
        "A sailboat on a calm lake surrounded by pine trees",
        "A modern minimalist living room with a gray sofa and large windows",
        "Cherry blossom trees along a river in spring",
        "A barista pouring latte art into a white cup, close-up",
        "Two runners on a coastal trail at sunrise",
        "A lighthouse on a rocky cliff during a storm",
        "Portrait of a young woman with freckles, natural light, 85mm",
        "A wooden rowboat tied to a dock at a quiet lake",
        "Hot air balloons over a valley at sunrise",
        "A plate of kimchi fried rice with a fried egg on top",
        "An old library with tall wooden shelves and warm lamps, books without readable titles",
        "A puppy and a kitten sitting side by side on a rug",
        "A waterfall in a lush green jungle",
        "Street portrait of a musician playing a saxophone",
        "A field of sunflowers under a bright summer sky",
        "Close-up of hands kneading bread dough on a floured table",
        "A snowy owl perched on a branch in winter",
        "A bowl of fresh berries with mint leaves, natural light",
        "Traditional Korean palace courtyard on a clear day",
        "A cyclist riding along a country road between rice fields",
        "Glass of iced tea with lemon slices on a wooden table outdoors",
        "Herd of deer in a forest clearing at dawn",
        "A grandmother and granddaughter cooking together in a kitchen",
        "Night sky full of stars above a mountain lake",
        "A red fox in fresh snow, wildlife photo",
        "A small fishing village harbor with colorful boats",
        "A pair of running shoes on a track, product photo without logos",
    ]
    assert len(general) == 50
    with (ROOT / "general_prompts.jsonl").open("w", encoding="utf-8") as f:
        for i, p in enumerate(general):
            f.write(json.dumps(dict(id=f"gen-{i:02d}", prompt_en=p), ensure_ascii=False) + "\n")
    n_flag = sum(r["has_unseen_syllable"] for r in rows)
    print(f"prompts={len(rows)} with_unseen_syllable={n_flag} usyl={len(srows)} general={len(general)}")


if __name__ == "__main__":
    main()
