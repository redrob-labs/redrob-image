"""Korean text for synthetic images: corpus fragments, signage templates, and a coverage sampler.

Three sources feed the per-image plan:
  (a) corpus  - Korean Wikipedia sentences and word stems (data/corpus/, CC BY-SA 4.0)
  (b) templates - store names, prices, notices, menu items, book titles, product labels, posters
  (c) coverage - a dataset-level pass that tops up every KS X 1001 syllable to >= N occurrences
      (stage 1) and, with stage 2, every other modern syllable to >= N2 occurrences, except the
      frozen unseen syllables. Real corpus words containing the syllable are used first; what the
      corpus never produces goes into short name-like or nonsense 2-4 syllable words.
Every string passes UnseenFilter before it is accepted.
"""
from __future__ import annotations

import collections
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "fonts"))
from fetch_fonts import ksx1001_syllables  # noqa: E402
from unseen import UnseenFilter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CDIR = ROOT / "data" / "corpus"
KS = ksx1001_syllables()
KS_SET = frozenset(KS)
ALL_SYL = [chr(c) for c in range(0xAC00, 0xD7A4)]

SCENES = ["sign", "poster", "book_cover", "product_label", "menu", "notice"]

SURNAMES = list("김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민진나지엄채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국어은편용")
SHOP_TYPES = ["식당", "카페", "분식", "마트", "약국", "미용실", "서점", "정육점", "베이커리", "치킨", "국밥", "꽃집",
              "세탁소", "문구", "철물점", "안경원", "치과", "한의원", "부동산", "편의점", "떡집", "반찬가게", "수산",
              "커피", "학원", "피아노", "태권도", "헬스", "사진관", "공방", "갤러리", "책방", "슈퍼", "상회", "닭갈비",
              "냉면", "칼국수", "만두", "김밥", "족발", "횟집", "포차", "제과점", "과일가게", "건강원"]
SHOP_PREFIX = ["행복", "우리", "새", "착한", "옛날", "원조", "할매", "엄마손", "솔", "한빛", "푸른", "다온", "온새미",
               "하늘", "바다", "산들", "별빛", "햇살", "봄날", "가람", "누리", "미소", "소망", "청춘", "골목", "동네",
               "명품", "참", "진", "큰", "작은", "제일", "으뜸", "백년", "대박", "황금"]
FOODS = ["김치찌개", "된장찌개", "순두부찌개", "부대찌개", "비빔밥", "불고기", "제육볶음", "떡볶이", "순대", "튀김",
         "김밥", "라면", "냉면", "물냉면", "비빔냉면", "칼국수", "수제비", "잔치국수", "콩국수", "만두", "갈비탕",
         "설렁탕", "삼계탕", "육개장", "돈가스", "오므라이스", "카레라이스", "볶음밥", "짜장면", "짬뽕", "탕수육",
         "아메리카노", "카페라테", "바닐라라테", "녹차라테", "자몽에이드", "레몬에이드", "유자차", "생강차",
         "식혜", "수정과", "팥빙수", "호떡", "붕어빵", "계란빵", "닭강정", "양념치킨", "후라이드", "족발", "보쌈",
         "해물파전", "김치전", "감자전", "막국수", "쌀국수", "우동", "초밥", "회덮밥", "오징어볶음", "고등어구이"]
PRODUCTS = ["사과", "배", "감귤", "딸기", "포도", "복숭아", "고구마", "감자", "양파", "당근", "쌀", "현미", "찹쌀",
            "참기름", "들기름", "간장", "된장", "고추장", "꿀", "녹차", "보리차", "두부", "우유", "요거트", "치즈",
            "김", "미역", "멸치", "새우젓", "고춧가루", "참깨", "견과", "누룽지", "쌀과자", "약과", "한과", "젤리",
            "비누", "샴푸", "핸드크림", "세제", "물티슈", "수세미", "생수", "탄산수", "주스"]
PRODUCT_ADJ = ["유기농", "국내산", "무농약", "프리미엄", "저염", "수제", "친환경", "햇", "신선한", "전통", "명품",
               "순수", "천연", "착한", "알뜰"]
UNITS = ["500g", "1kg", "2kg", "300g", "250ml", "500ml", "1L", "1.5L", "10개입", "20개입", "3개", "1박스", "1봉"]
NOTICE_TITLES = ["안내", "공지사항", "알림", "주의", "출입 금지", "주차 금지", "임시 휴업", "영업 안내", "휴무 안내",
                 "이용 안내", "금연 구역", "관계자 외 출입 금지", "공사 중", "고객 안내", "분실물 보관"]
NOTICE_LINES = ["영업시간 오전 {h1}시 ~ 오후 {h2}시", "매주 {d}요일 정기 휴무", "{m}월 {dd}일부터 {m2}월 {dd2}일까지 휴업합니다",
                "이곳은 사유지입니다", "차량 진입을 삼가 주세요", "소지품은 각자 관리해 주세요", "엘리베이터 점검 중입니다",
                "화장실은 2층에 있습니다", "반려동물 동반 가능", "현금 결제 시 10% 할인", "카드 결제 가능합니다",
                "문의 {tel}", "배달 가능 지역 문의", "{n}층 이용해 주세요", "계단 이용 시 주의하세요",
                "쓰레기 무단 투기 금지", "조용히 해 주세요", "손을 깨끗이 씻어 주세요", "마스크를 착용해 주세요",
                "예약 손님 우선입니다", "오늘의 재료가 모두 소진되었습니다", "신규 회원 모집 중", "택배는 경비실에 맡겨 주세요"]
DAYS = list("월화수목금토일")
POSTER_KIND = ["축제", "음악회", "전시회", "박람회", "공연", "체험전", "강연", "바자회", "마라톤", "영화제", "문화제",
               "장터", "벼룩시장", "사진전", "독서회", "설명회", "발표회", "콘서트", "페스티벌", "대회"]
SEASONS = ["봄", "여름", "가을", "겨울", "새해", "한가위", "가정의 달", "크리스마스"]
PLACES = ["시민회관", "문화센터", "중앙공원", "대강당", "구민회관", "도서관", "체육관", "광장", "아트홀", "전시관",
          "주민센터", "강변 공원", "야외무대", "소극장"]
PUBLISHERS = ["출판사", "북스", "문고", "서원", "책방", "프레스", "미디어", "출판"]
BOOK_PATTERNS = ["{a}의 {b}", "{a}와 {b}", "{a}에서 {b}까지", "나의 {a}", "{a} 이야기", "우리가 {a}", "{a}을 위하여",
                 "다시 {a}", "{a}의 시간", "{a}에 대하여", "{a}, 그리고 {b}", "{a} 수업", "처음 만나는 {a}",
                 "{a} 입문", "{a}의 역사", "작은 {a}", "{a}의 기술"]
MENU_TITLES = ["메뉴", "오늘의 메뉴", "차림표", "MENU", "음료", "식사", "점심 특선", "세트 메뉴", "디저트", "추천 메뉴"]


def fmt_price(rng: random.Random) -> str:
    v = rng.choice([1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000,
                    9500, 10000, 11000, 12000, 13000, 15000, 18000, 20000, 25000, 30000, 35000])
    s = f"{v:,}"
    return rng.choice([f"{s}원", f"{s}", f"₩{s}", f"{v // 1000}.{(v % 1000) // 100}"] if v % 100 == 0 else [f"{s}원"])


def fmt_tel(rng: random.Random) -> str:
    a = rng.choice(["02", "031", "032", "051", "053", "042", "062", "064", "010", "1588", "070"])
    if a == "1588":
        return f"1588-{rng.randint(1000, 9999)}"
    return f"{a}-{rng.randint(200, 9999)}-{rng.randint(1000, 9999)}"


class TextSource:
    def __init__(self, filt: UnseenFilter | None = None):
        self.f = filt or UnseenFilter.load()
        sents = [x for x in (CDIR / "sentences.txt").read_text(encoding="utf-8").splitlines() if '"' not in x]
        self.sentences_short = [s for s in sents if len(s) <= 22]
        self.sentences_long = [s for s in sents if 22 < len(s) <= 40]
        self.words_all: list[tuple[str, int]] = []
        for ln in (CDIR / "words.txt").read_text(encoding="utf-8").splitlines():
            w, c = ln.split("\t")
            self.words_all.append((w, int(c)))
        self.nouns = [w for w, c in self.words_all if c >= 5 and 2 <= len(w) <= 4 and not w.endswith(("다", "고", "며", "서", "지", "게"))]
        self._syl_words: dict[str, list[str]] | None = None

    # ---------------- natural strings ----------------
    def _ok(self, s: str) -> bool:
        return self.f.ok(s)

    def pick(self, rng: random.Random, gen) -> str:
        for _ in range(50):
            s = gen(rng)
            if s and self._ok(s):
                return s
        raise RuntimeError("could not produce a clean string")

    def noun(self, rng):
        return rng.choice(self.nouns)

    def shop_name(self, rng):
        r = rng.random()
        if r < 0.4:
            return f"{rng.choice(SHOP_PREFIX)}{rng.choice(SHOP_TYPES)}"
        if r < 0.7:
            return f"{self.noun(rng)} {rng.choice(SHOP_TYPES)}"
        if r < 0.85:
            return f"{rng.choice(SURNAMES)}{rng.choice(['씨네', '가네', '네'])} {rng.choice(SHOP_TYPES)}"
        return f"{rng.choice(SHOP_PREFIX)} {self.noun(rng)}"

    def gen(self, role: str, rng: random.Random) -> str:
        return self.pick(rng, lambda r: self._gen(role, r))

    def _gen(self, role: str, rng: random.Random) -> str:
        if role == "shop":
            return self.shop_name(rng)
        if role == "slogan":
            return rng.choice([lambda: rng.choice(self.sentences_short),
                               lambda: f"since {rng.randint(1965, 2024)}",
                               lambda: f"{rng.choice(['정성을 담은', '매일 신선한', '맛있는', '따뜻한', '건강한', '정직한'])} {self.noun(rng)}",
                               lambda: fmt_tel(rng),
                               lambda: f"{rng.choice(['본점', '2호점', '직영점', '역전점', '시장점'])}"])()
        if role == "poster_title":
            return rng.choice([lambda: f"{rng.choice(SEASONS)} {rng.choice(POSTER_KIND)}",
                               lambda: f"제{rng.randint(1, 40)}회 {self.noun(rng)} {rng.choice(POSTER_KIND)}",
                               lambda: f"{self.noun(rng)} {rng.choice(POSTER_KIND)}",
                               lambda: f"{rng.randint(2025, 2027)} {self.noun(rng)} {rng.choice(POSTER_KIND)}"])()
        if role == "poster_when":
            m = rng.randint(1, 12)
            return rng.choice([f"{m}월 {rng.randint(1, 28)}일 ({rng.choice(DAYS)}) 오후 {rng.randint(1, 8)}시",
                               f"{rng.randint(2025, 2027)}. {m}. {rng.randint(1, 28)} - {m}. {rng.randint(1, 28)}",
                               f"매주 {rng.choice(DAYS)}요일 저녁 {rng.randint(6, 9)}시",
                               f"{m}월 {rng.randint(1, 28)}일 ~ {m}월 {rng.randint(1, 28)}일"])
        if role == "poster_where":
            return rng.choice([f"장소: {self.noun(rng)} {rng.choice(PLACES)}", f"{rng.choice(PLACES)} {rng.randint(1, 5)}층",
                               f"{self.noun(rng)} {rng.choice(PLACES)}", "입장료 무료", f"참가비 {fmt_price(rng)}"])
        if role == "sentence":
            return rng.choice(self.sentences_short)
        if role == "sentence_long":
            return rng.choice(self.sentences_long)
        if role == "book_title":
            p = rng.choice(BOOK_PATTERNS)
            return p.format(a=self.noun(rng), b=self.noun(rng))
        if role == "author":
            name = rng.choice(SURNAMES) + "".join(rng.choice(KS[::7]) for _ in range(rng.choice([1, 2, 2])))
            return rng.choice([f"{name} 지음", f"{name} 장편소설", f"{name} 글", f"{name} 엮음", f"{name} 옮김", name])
        if role == "publisher":
            return f"{self.noun(rng)}{rng.choice(PUBLISHERS)}"
        if role == "product":
            return rng.choice([f"{rng.choice(PRODUCT_ADJ)} {rng.choice(PRODUCTS)}", rng.choice(PRODUCTS),
                               f"{self.noun(rng)} {rng.choice(PRODUCTS)}"])
        if role == "spec":
            return rng.choice([rng.choice(UNITS), f"원산지: {rng.choice(['국내산', '수입산', '제주', '강원도', '전라남도', '경상북도'])}",
                               f"유통기한 {rng.randint(2026, 2028)}.{rng.randint(1, 12):02d}.{rng.randint(1, 28):02d}",
                               f"{rng.choice(UNITS)} / {rng.choice(['냉장', '냉동', '실온'])} 보관", "무첨가 무색소"])
        if role == "price":
            return rng.choice([fmt_price(rng), f"특가 {fmt_price(rng)}", f"{rng.randint(1, 5)}+1", f"{rng.randint(10, 50)}% 할인"])
        if role == "menu_title":
            return rng.choice(MENU_TITLES + [self.shop_name(rng)])
        if role == "menu_items":
            n = rng.randint(3, 6)
            items = rng.sample(FOODS, n)
            return "\n".join(f"{it}  {fmt_price(rng)}" for it in items)
        if role == "notice_title":
            return rng.choice(NOTICE_TITLES)
        if role == "notice_lines":
            n = rng.randint(1, 3)
            lines = []
            for t in rng.sample(NOTICE_LINES, n):
                h1 = rng.randint(7, 11)
                lines.append(t.format(h1=h1, h2=rng.randint(6, 11), d=rng.choice(DAYS), m=rng.randint(1, 12),
                                      dd=rng.randint(1, 28), m2=rng.randint(1, 12), dd2=rng.randint(1, 28),
                                      tel=fmt_tel(rng), n=rng.randint(1, 5)))
            if rng.random() < 0.3:
                lines.append(rng.choice(self.sentences_short))
            return "\n".join(lines)
        if role == "org":
            return rng.choice([f"{self.noun(rng)} 관리사무소", f"{self.noun(rng)}구청", "관리자 일동", f"{self.shop_name(rng)} 대표",
                               f"{rng.randint(2025, 2027)}년 {rng.randint(1, 12)}월 {rng.randint(1, 28)}일"])
        raise KeyError(role)

    SCENE_ROLES = {
        "sign": (["shop"], ["slogan", "slogan", "sentence"]),
        "poster": (["poster_title", "poster_when"], ["poster_where", "sentence"]),
        "book_cover": (["book_title", "author"], ["publisher", "sentence"]),
        "product_label": (["product", "price"], ["spec", "spec"]),
        "menu": (["menu_title", "menu_items"], ["slogan", "sentence"]),
        "notice": (["notice_title", "notice_lines"], ["org", "sentence_long"]),
    }

    def scene_blocks(self, scene: str, rng: random.Random) -> list[dict]:
        req, opt = self.SCENE_ROLES[scene]
        roles = list(req)
        k = rng.randint(0, min(len(opt), 4 - len(req)))
        roles += rng.sample(opt, k)
        return [{"role": r, "text": self.gen(r, rng)} for r in roles]

    # ---------------- coverage ----------------
    def syllable_words(self) -> dict[str, list[str]]:
        """syllable -> clean corpus words (2-5 syllables) containing it, most frequent first."""
        if self._syl_words is None:
            d: dict[str, list[str]] = collections.defaultdict(list)
            for w, c in self.words_all:
                if 2 <= len(w) <= 5 and c >= 1:
                    for ch in set(w):
                        if len(d[ch]) < 40:
                            d[ch].append(w)
            self._syl_words = d
        return self._syl_words

    def coverage_words(self, deficit: dict[str, int], rng: random.Random, filler: list[str]) -> list[str]:
        """Words whose syllables bring every deficit syllable up by its missing count.

        deficit: syllable -> occurrences still missing. Real corpus words are used first (cycling
        through different words); the rest is packed into name-like / nonsense words of 2-4
        syllables, mixing deficit syllables with common filler syllables.
        """
        need = dict((s, n) for s, n in deficit.items() if n > 0 and not self.f.bad_syllable(s))
        out: list[str] = []
        sw = self.syllable_words()
        for s in sorted(need, key=lambda x: (need[x], x)):
            if s not in KS_SET:
                continue  # corpus words with rare syllables are almost always typos (대한민굮)
            # only words made entirely of KS X 1001 syllables, so no typo words slip in
            cands = [w for w in sw.get(s, []) if all(ch in KS_SET for ch in w) and self._ok(w)]
            i = 0
            while need.get(s, 0) > 0 and cands and i < need[s] + 5:
                w = cands[i % len(cands)]
                i += 1
                out.append(w)
                for ch in w:
                    if ch in need:
                        need[ch] -= 1
                if i >= 3 * len(cands):
                    break
        pool = [s for s, n in need.items() for _ in range(max(0, n))]
        rng.shuffle(pool)
        while pool:
            k = rng.choice([2, 3, 3, 4])
            chunk, pool = pool[:k], pool[k:]
            form = rng.random()
            for _ in range(30):
                if form < 0.3 and len(chunk) <= 3:
                    w = rng.choice(SURNAMES) + "".join(chunk)  # name-like: 김 + 2-3 syllables
                elif form < 0.4 and len(chunk) <= 3:
                    w = rng.choice(filler) + "".join(chunk)
                else:
                    w = "".join(chunk) if len(chunk) >= 2 else chunk[0] + rng.choice(filler)
                if self._ok(w):
                    break
                rng.shuffle(chunk)
            else:
                # Each syllable is clean by itself; emit them one per word as a last resort.
                for ch in chunk:
                    out.append(ch + rng.choice(filler))
                continue
            out.append(w)
        return out


FILLER = list("가나다라마바사아자차카타파하고노도로모보소오조호구누두루무부수우주후이리미비시지기니디")


def coverage_block_text(scene: str, words: list[str], rng: random.Random) -> tuple[str, str]:
    """Wrap coverage words in scene-appropriate context. Returns (role, text)."""
    if scene == "menu":
        return "cov_items", "\n".join(f"{w}  {fmt_price(rng)}" for w in words)
    if scene == "product_label":
        return "cov_product", " ".join(words) + rng.choice(["", f" {rng.choice(UNITS)}"])
    if scene == "book_cover":
        return "cov_author", " · ".join(words) + rng.choice([" 지음", " 엮음", ""])
    if scene == "sign":
        return "cov_tag", " ".join(words)
    if scene == "notice":
        return "cov_line", rng.choice(["담당: ", "문의: ", "참고: ", ""]) + ", ".join(words)
    return "cov_tag", " ".join(words)
