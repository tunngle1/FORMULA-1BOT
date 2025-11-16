import os
import logging
import html
import time
from datetime import datetime, timezone, timedelta

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # для старых Python

import requests
from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# ========= НАСТРОЙКИ =========

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7407100700:AAGjSu04_uYVcjuYagWBb5aWEbkLqqWJXfA")

OPENF1_BASE_URL = "https://api.openf1.org/v1"
F1API_BASE_URL = "https://f1api.dev"

# Дополнительный источник для таблиц чемпионата (GraphQL)
F1_GRAPHQL_ENDPOINT = "https://f1-graphql.davideladisa.it/graphql"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_msk_tz():
    """
    Пытаемся взять настоящую таймзону Europe/Moscow (если установлена tzdata).
    Если её нет (часто на Windows), используем фиксированный UTC+3.
    """
    if ZoneInfo is not None:
        try:
            return ZoneInfo("Europe/Moscow")
        except Exception:
            pass
    return timezone(timedelta(hours=3))


MSK_TZ = get_msk_tz()

# ========= ПРОСТОЕ "ХРАНИЛИЩЕ" В ПАМЯТИ =========

leagues: dict[int, dict] = {}

# leagues[chat_id] = {
#   "year": int,
#   "meeting": None,
#   "drivers": [],
#   "qual_results": [],
#   "phase": "IDLE",
#   "bets_q1": {},
#   "bets": {meeting_key: {...}},
#   "xp": {user_id: points},
#   "pending_bets": {user_id: {...}},
# }

# Кэш для сезонных таблиц: {year: (driver_stats, team_stats)}
SEASON_CACHE: dict[int, tuple[dict, dict]] = {}

# Кэш календаря
F1API_RACES_CACHE: dict[int, list[dict]] = {}
F1API_RACES_BY_ID: dict[str, dict] = {}

# Настройки ставок
BET_MAX = {
    "Q1": 5,
    "Q2": 5,
    "Q3": 3,
    "SPRINT": 3,
    "RACE": 3,
}

# Для вылетов (Q1, Q2) — очки за попадание
BET_POINTS_ELIM = {
    "Q1": 1,
    "Q2": 1,
}

# Для топ-3 — базовые и за точное место
TOP3_POINTS = {
    "Q3": {"in_top": 1, "exact": 2},
    "SPRINT": {"in_top": 2, "exact": 3},
    "RACE": {"in_top": 3, "exact": 5},
}

# какие сессии считаем важными для ставок/таймера
RELEVANT_F1API_SESSIONS = [
    ("Qualifying", "qualy", 60),        # 60 минут на квулу
    ("Sprint Race", "sprintRace", 60),
    ("Race", "race", 120),              # до 2 часов на гонку
]


# ========= ПРОСТЫЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ HTTP =========

def openf1_get(path: str, params: dict | None = None, *, retries: int = 3, backoff: float = 0.7):
    """
    Обёртка над OpenF1 с простым retry на 429 Too Many Requests.
    """
    url = OPENF1_BASE_URL + path
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        resp = requests.get(url, params=params, timeout=15)

        if resp.status_code == 429:
            wait = backoff * attempt
            logger.warning(
                "OpenF1 429 on %s params=%s, attempt %d/%d, sleep %.1fs",
                path, params, attempt, retries, wait,
            )
            last_exc = requests.HTTPError(
                f"429 Too Many Requests for {url}", response=resp
            )
            time.sleep(wait)
            continue

        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            last_exc = e
            break

        return resp.json()

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"OpenF1 request failed for {url}")


def f1api_get(path: str, params: dict | None = None):
    """Простая обёртка для f1api.dev."""
    url = F1API_BASE_URL + path
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def f1_graphql_query(query: str, variables: dict | None = None) -> dict:
    """
    Вызов F1 GraphQL.
    """
    payload: dict = {"query": query}
    if variables:
        payload["variables"] = variables
    resp = requests.post(F1_GRAPHQL_ENDPOINT, json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")
    return data.get("data", {})


def load_f1api_races(year: int) -> list[dict]:
    """
    Загружает календарь сезона:
    - для текущего года сначала пробуем /api/current
    - затем /api/{year}
    Результат кладём в кэш и индекс по raceId.
    """
    if year in F1API_RACES_CACHE:
        return F1API_RACES_CACHE[year]

    logger.info("load_f1api_races: requesting calendar for year %s", year)

    now_year = datetime.now(timezone.utc).year
    paths = []
    if year == now_year:
        paths.append("/api/current")
    paths.append(f"/api/{year}")

    data = None
    last_exc: Exception | None = None
    for p in paths:
        try:
            data = f1api_get(p)
            logger.info("load_f1api_races: got response from %s", p)
            break
        except Exception as e:
            last_exc = e
            logger.warning("f1api.dev error on %s: %s", p, e)

    if data is None:
        raise RuntimeError(
            f"Не удалось загрузить календарь сезона {year}: {last_exc}"
        )

    races_data = data.get("races") or data.get("race") or []
    if isinstance(races_data, list):
        races_list = races_data
    elif isinstance(races_data, dict):
        races_list = [races_data]
    else:
        races_list = []

    for r in races_list:
        if "season" not in r:
            r["season"] = data.get("season", year)
        race_id = r.get("raceId") or f"{r.get('season', year)}_{r.get('round')}"
        if race_id:
            F1API_RACES_BY_ID[race_id] = r

    logger.info(
        "load_f1api_races: loaded %d races for year %s",
        len(races_list),
        year,
    )

    F1API_RACES_CACHE[year] = races_list
    return races_list


def get_or_create_league(chat_id: int) -> dict:
    if chat_id not in leagues:
        leagues[chat_id] = {
            "year": datetime.now(timezone.utc).year,
            "meeting": None,
            "drivers": [],
            "qual_results": [],
            "phase": "IDLE",
            "bets_q1": {},
            "bets": {},
            "xp": {},
            "pending_bets": {},
        }
    else:
        league = leagues[chat_id]
        league.setdefault("bets", {})
        league.setdefault("xp", {})
        league.setdefault("pending_bets", {})
    return leagues[chat_id]


def get_chat_league(chat_id: int) -> dict:
    return get_or_create_league(chat_id)


def find_latest_meeting(year: int):
    meetings = openf1_get("/meetings", {"year": year})
    if not meetings:
        return None
    meetings.sort(key=lambda m: m.get("date_start") or "")
    return meetings[-1]


def find_qual_session(meeting_key: int):
    sessions = openf1_get("/sessions", {"meeting_key": meeting_key})

    def lower(v):
        return str(v or "").lower()

    keywords = ["qualifying", "short qualifying"]

    for s in sessions:
        if any(kw in lower(s.get("session_type")) for kw in keywords):
            return s
    for s in sessions:
        if any(kw in lower(s.get("session_name")) for kw in keywords):
            return s
    return None


def sort_results_by_position(results: list[dict]) -> list[dict]:
    """Сортируем: сначала с позицией, потом без."""
    def key(r: dict):
        pos = r.get("position")
        if isinstance(pos, int):
            return (0, pos)
        return (1, r.get("driver_number") or 999)

    return sorted(results, key=key)


def split_qual_results(results: list[dict]):
    """Возвращает (q1_out, q2_out, q3_top3) по позициям."""
    sorted_res = sort_results_by_position(results)
    q3_top3 = [
        r for r in sorted_res
        if isinstance(r.get("position"), int) and 1 <= r["position"] <= 3
    ]
    q2_out = [
        r for r in sorted_res
        if isinstance(r.get("position"), int) and 11 <= r["position"] <= 15
    ]
    q1_out = [
        r for r in sorted_res
        if isinstance(r.get("position"), int) and 16 <= r["position"] <= 20
    ]
    return q1_out, q2_out, q3_top3


def parse_acronym_input(text: str) -> list[str]:
    """Парсим строку вида 'ver, nor ham' в ['VER','NOR','HAM']"""
    tokens = [t.strip().upper() for t in text.replace("\n", " ").split() if t.strip()]
    cleaned = []
    seen = set()
    for t in tokens:
        t = t.strip(",; ")
        if not t:
            continue
        if len(t) > 3:
            t = t[:3]
        if t not in seen:
            seen.add(t)
            cleaned.append(t)
    return cleaned[:5]


def dedupe_drivers(drivers: list[dict]) -> list[dict]:
    """Убираем дубли пилотов (берём по driver_number)."""
    by_num = {}
    for d in drivers:
        num = d.get("driver_number")
        if num is None:
            continue
        if num not in by_num:
            by_num[num] = d
    return list(by_num.values())


def format_seconds_like_laptime(value) -> str:
    """Форматируем секунды в вид 1:23.456 / 23.456 / 1:02:03.123."""
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            sec = float(value)
        except ValueError:
            return value
    else:
        try:
            sec = float(value)
        except (TypeError, ValueError):
            return str(value)

    if sec < 0:
        sec = -sec

    ms = int(round((sec - int(sec)) * 1000))
    total_seconds = int(sec)
    s = total_seconds % 60
    total_minutes = total_seconds // 60
    m = total_minutes % 60
    h = total_minutes // 60

    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}.{ms:03d}"
    elif total_minutes > 0:
        return f"{total_minutes:d}:{s:02d}.{ms:03d}"
    else:
        return f"{s:d}.{ms:03d}"


def extract_last_segment(value):
    """
    Для полей, которые в квале могут быть массивом [Q1,Q2,Q3]:
    берём последнее ненулевое значение.
    Для одиночного числа — возвращаем как есть.
    """
    if value is None:
        return None
    if isinstance(value, list):
        non_null = [v for v in value if v is not None]
        if not non_null:
            return None
        return non_null[-1]
    return value


def pre_block(lines: list[str]) -> str:
    """HTML <pre> с экранированием — красиво ровные столбцы."""
    return "<pre>" + "\n".join(html.escape(line) for line in lines) + "</pre>"


def parse_iso_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


# ========= UTC → МСК ДЛЯ F1API + COUNTER =========

def parse_utc_to_msk_dt(date_str: str | None, time_str: str | None) -> datetime | None:
    """
    Превращает дату/время из календаря (UTC) в datetime в МСК.
    """
    if not date_str and not time_str:
        return None
    if not date_str:
        return None
    t_str = time_str or "00:00:00"
    raw = f"{date_str}T{t_str}"
    if raw.endswith("Z"):
        raw = raw.replace("Z", "+00:00")
    else:
        raw = raw + "+00:00"
    try:
        dt_utc = datetime.fromisoformat(raw)
    except Exception:
        return None
    return dt_utc.astimezone(MSK_TZ)


def format_dt_msk(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


def format_countdown(target_dt: datetime) -> str:
    """
    Красивый текст обратного отсчёта до target_dt (МСК).
    """
    now = datetime.now(MSK_TZ)
    delta = target_dt - now
    total = int(delta.total_seconds())
    if total <= 0:
        return "уже началось или прошло"

    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)

    parts = []
    if days > 0:
        parts.append(f"{days}д")
    if hours > 0 or days > 0:
        parts.append(f"{hours}ч")
    if mins > 0 or hours > 0 or days > 0:
        parts.append(f"{mins}м")
    else:
        parts.append(f"{secs}с")

    return " ".join(parts)


def get_current_relevant_session(year: int | None = None) -> dict | None:
    """
    Возвращает текущую важную сессию (Qualifying / Sprint / Race),
    если она сейчас идёт по МСК.
    """
    now = datetime.now(MSK_TZ)
    if year is None:
        year = now.year
    try:
        races = load_f1api_races(year)
    except Exception:
        return None

    current = None
    for r in races:
        sch = r.get("schedule") or {}
        for label, key, duration_min in RELEVANT_F1API_SESSIONS:
            s = sch.get(key) or {}
            dt = parse_utc_to_msk_dt(s.get("date"), s.get("time"))
            if not dt:
                continue
            end = dt + timedelta(minutes=duration_min)
            if dt <= now <= end:
                if current is None or dt > current["start"]:
                    current = {
                        "label": label,
                        "key": key,
                        "start": dt,
                        "end": end,
                        "race": r,
                    }
    return current


def get_next_relevant_session(year: int | None = None) -> dict | None:
    """
    Возвращает ближайшую будущую важную сессию (Qualifying / Sprint / Race).
    """
    now = datetime.now(MSK_TZ)
    if year is None:
        year = now.year
    try:
        races = load_f1api_races(year)
    except Exception:
        return None

    best = None
    for r in races:
        sch = r.get("schedule") or {}
        for label, key, duration_min in RELEVANT_F1API_SESSIONS:
            s = sch.get(key) or {}
            dt = parse_utc_to_msk_dt(s.get("date"), s.get("time"))
            if not dt:
                continue
            if dt <= now:
                continue
            if best is None or dt < best["start"]:
                best = {
                    "label": label,
                    "key": key,
                    "start": dt,
                    "race": r,
                    "duration_min": duration_min,
                }
    return best


def is_bet_window_open(bet_type: str, league: dict) -> bool:
    """
    Проверяем, открыто ли окно ставок (первые 5 минут нужной сессии)
    для переданного типа ставки.
    """
    curr = get_current_relevant_session(league["year"])
    if not curr:
        return False

    now = datetime.now(MSK_TZ)
    start = curr["start"]
    if now > start + timedelta(minutes=5):
        return False

    key = curr["key"]

    if bet_type in ("Q1", "Q2", "Q3") and key == "qualy":
        return True
    if bet_type == "SPRINT" and key == "sprintRace":
        return True
    if bet_type == "RACE" and key == "race":
        return True

    return False


# ========= F1 GRAPHQL: СЕЗОННЫЕ ТАБЛИЦЫ =========

def compute_season_standings_from_graphql(year: int):
    """
    Получаем таблицу пилотов и конструкторов через GraphQL.
    """

    # --- ЛИЧНЫЙ ЗАЧЁТ ---
    driver_query = """
    query DriverStandings($season: Int!) {
      findManyDriverStanding(
        where: { year: { equals: $season } }
        orderBy: [{ points: desc }, { position: asc }]
      ) {
        position
        points
        wins
        driver {
          id
          code
          permanentNumber
          fullName
          firstName
          lastName
        }
        constructor {
          name
        }
      }
    }
    """

    drv_data = f1_graphql_query(driver_query, {"season": year})
    drv_items = drv_data.get("findManyDriverStanding") or []

    driver_stats: dict[int | str, dict] = {}

    for item in drv_items:
        pos = item.get("position")
        pts = item.get("points") or 0
        wins = item.get("wins") or 0
        d = item.get("driver") or {}
        cons = item.get("constructor") or {}

        num = d.get("permanentNumber")
        if num is None:
            num = d.get("id") or d.get("code") or f"drv_{pos}"

        try:
            pts_f = float(pts)
        except (TypeError, ValueError):
            pts_f = 0.0

        full_name = d.get("fullName") or d.get("name") or (
            (d.get("firstName") or "") + " " + (d.get("lastName") or "")
        ).strip() or f"Driver {num}"

        ac = (d.get("code") or "").upper() or "???"
        team = cons.get("name") or "Unknown"

        driver_stats[num] = {
            "driver_number": num,
            "full_name": full_name,
            "acronym": ac,
            "last_team": team,
            "points": pts_f,
            "wins": wins,
        }

    # --- КУБОК КОНСТРУКТОРОВ ---
    team_query = """
    query ConstructorStandings($season: Int!) {
      findManyConstructorStanding(
        where: { year: { equals: $season } }
        orderBy: [{ points: desc }, { position: asc }]
      ) {
        position
        points
        constructor {
          name
        }
      }
    }
    """

    team_data = f1_graphql_query(team_query, {"season": year})
    team_items = team_data.get("findManyConstructorStanding") or []

    team_stats: dict[str, dict] = {}
    for item in team_items:
        cons = item.get("constructor") or {}
        name = cons.get("name") or "Unknown"
        pts = item.get("points") or 0
        try:
            pts_f = float(pts)
        except (TypeError, ValueError):
            pts_f = 0.0

        team_stats[name] = {
            "team_name": name,
            "points": pts_f,
        }

    logger.info(
        "GraphQL standings: year %s, drivers=%d, teams=%d",
        year,
        len(driver_stats),
        len(team_stats),
    )

    return driver_stats, team_stats


# ========= ВСПОМОГАТЕЛЬНЫЕ ДЛЯ СЕЗОННЫХ ТАБЛИЦ (OpenF1 fallback) =========

def compute_season_standings_from_openf1(year: int):
    """
    Старая логика: считаем очки по всем гонкам сезона.
    Используем как fallback, если GraphQL не сработал.
    """
    try:
        meetings = openf1_get("/meetings", {"year": year})
    except Exception as e:
        logger.exception("compute_season_standings_from_openf1: meetings error")
        raise RuntimeError(f"Ошибка загрузки этапов сезона {year}: {e}")

    meeting_keys = {m["meeting_key"] for m in meetings if m.get("meeting_key") is not None}

    try:
        sessions_all = openf1_get("/sessions", {"year": year})
    except Exception as e:
        logger.exception("compute_season_standings_from_openf1: sessions(year) error")
        raise RuntimeError(f"Ошибка загрузки сессий сезона {year}: {e}")

    def lower(v):
        return str(v or "").lower()

    race_sessions = [
        s for s in sessions_all
        if s.get("meeting_key") in meeting_keys
        and "race" in lower(s.get("session_type"))
    ]

    try:
        drivers_raw = openf1_get("/drivers", {"session_key": "latest"})
    except Exception as e:
        logger.exception("compute_season_standings_from_openf1: drivers(latest) error")
        drivers_raw = []

    driver_info_by_num: dict[int, dict] = {}
    for d in drivers_raw:
        num = d.get("driver_number")
        if num is None:
            continue
        driver_info_by_num[num] = d

    driver_stats: dict[int, dict] = {}
    team_stats: dict[str, dict] = {}

    for s in race_sessions:
        session_key = s.get("session_key")
        if session_key is None:
            continue
        try:
            results = openf1_get("/session_result", {"session_key": session_key})
        except Exception:
            logger.exception("compute_season_standings_from_openf1: session_result error")
            continue

        for r in results:
            num = r.get("driver_number")
            if num is None:
                continue

            dsq = r.get("dsq")
            dns = r.get("dns")

            points = r.get("points")
            if points is None:
                pos = r.get("position")
                pts_map = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
                if isinstance(pos, int) and 1 <= pos <= 10 and not dsq and not dns:
                    points = pts_map[pos - 1]
                else:
                    points = 0

            try:
                pts_f = float(points)
            except (TypeError, ValueError):
                pts_f = 0.0

            info = driver_info_by_num.get(num, {})
            full_name = info.get("full_name") or f"Driver #{num}"
            ac = (info.get("name_acronym") or "").upper() or "???"
            team = r.get("team_name") or info.get("team_name") or "Unknown"

            dstat = driver_stats.setdefault(
                num,
                {
                    "driver_number": num,
                    "full_name": full_name,
                    "acronym": ac,
                    "last_team": team,
                    "points": 0.0,
                    "wins": 0,
                },
            )
            dstat["points"] += pts_f
            if team:
                dstat["last_team"] = team

            pos = r.get("position")
            if isinstance(pos, int) and pos == 1 and pts_f > 0:
                dstat["wins"] += 1

            tstat = team_stats.setdefault(
                team,
                {
                    "team_name": team,
                    "points": 0.0,
                },
            )
            tstat["points"] += pts_f

    return driver_stats, team_stats


def compute_season_standings(year: int):
    """
    Главная функция: сначала пробуем GraphQL,
    если не получилось — считаем по результатам гонок.
    """
    if year in SEASON_CACHE:
        return SEASON_CACHE[year]

    try:
        driver_stats, team_stats = compute_season_standings_from_graphql(year)
        if driver_stats and team_stats:
            SEASON_CACHE[year] = (driver_stats, team_stats)
            logger.info("Season %s standings loaded from GraphQL", year)
            return driver_stats, team_stats
        else:
            logger.warning("GraphQL standings for year %s are empty, fallback", year)
    except Exception as e:
        logger.exception("compute_season_standings: GraphQL error, fallback: %s", e)

    driver_stats, team_stats = compute_season_standings_from_openf1(year)
    SEASON_CACHE[year] = (driver_stats, team_stats)
    logger.info("Season %s standings computed from race results", year)
    return driver_stats, team_stats


# ========= ХЕЛПЕРЫ ДЛЯ СТАВОК =========

def build_bet_keyboard(bet_type: str, league: dict, user_id: int):
    """Рисуем клавиатуру выбора пилотов для конкретного типа ставки."""
    pending = league.setdefault("pending_bets", {})
    state = pending.get(user_id)
    if not state or state.get("type") != bet_type:
        return None

    allowed_nums = state["allowed_drivers"]
    selected = set(state["selected"])
    drivers_all = {d["driver_number"]: d for d in league["drivers"]}

    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []

    for num in allowed_nums:
        d = drivers_all.get(num)
        if not d:
            continue
        ac = (d.get("name_acronym") or "").upper()
        label_base = f"{ac:>3} #{num:<2}"
        if num in selected:
            label = f"✅ {label_base}"
        else:
            label = label_base
        cb = f"bet_pick:{bet_type}:{num}"
        row.append(InlineKeyboardButton(label, callback_data=cb))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)

    cnt = len(selected)
    max_cnt = state["max_count"]

    rows.append(
        [InlineKeyboardButton(f"✅ Подтвердить ({cnt}/{max_cnt})", callback_data=f"bet_confirm:{bet_type}")]
    )
    rows.append(
        [
            InlineKeyboardButton("❌ Отмена", callback_data=f"bet_cancel:{bet_type}"),
            InlineKeyboardButton("⬅️ Назад", callback_data="menu_bets"),
        ]
    )

    return InlineKeyboardMarkup(rows)


async def open_bets_menu(query, league):
    """
    Открыть меню ставок для текущего Гран-при.
    """
    now = datetime.now(MSK_TZ)
    year = league["year"]

    current_sess = get_current_relevant_session(year)
    next_sess = get_next_relevant_session(year)

    # если ещё нет meeting — пробуем подтянуть
    if not league.get("meeting"):
        meeting = find_latest_meeting(year)
        if meeting:
            meeting_key = meeting["meeting_key"]
            qual_session = find_qual_session(meeting_key)
            if qual_session:
                drivers_raw = openf1_get("/drivers", {"meeting_key": meeting_key})
                drivers = dedupe_drivers(drivers_raw)
                try:
                    qual_results = openf1_get("/session_result", {"session_key": qual_session["session_key"]})
                except Exception:
                    qual_results = []

                league["meeting"] = meeting
                league["drivers"] = drivers
                league["qual_results"] = qual_results

    gp_name = league.get("meeting", {}).get("meeting_name") or "Текущий Гран-при"

    lines: list[str] = []
    lines.append(f"🎯 Ставки на текущий Гран-при")
    lines.append(gp_name)
    lines.append("")

    # состояние окон ставок
    q1_open = is_bet_window_open("Q1", league)
    q2_open = is_bet_window_open("Q2", league)
    q3_open = is_bet_window_open("Q3", league)
    sprint_open = is_bet_window_open("SPRINT", league)
    race_open = is_bet_window_open("RACE", league)

    # информация о лайве / ближайшем событии
    if current_sess and current_sess["key"] in ("qualy", "sprintRace", "race"):
        r = current_sess["race"]
        race_name = r.get("raceName") or r.get("circuit", {}).get("circuitName") or "Гран-при"
        label = current_sess["label"]
        start = current_sess["start"]
        end = current_sess["end"]

        lines.append(f"Сейчас идёт: {label} {race_name}")
        lines.append(f"Старт: {format_dt_msk(start)} МСК")
        lines.append(f"Окончание (по расписанию): {format_dt_msk(end)} МСК")
        lines.append("")
        lines.append("Ставки принимаются только в первые 5 минут соответствующей сессии.")
        lines.append("")
    else:
        lines.append("Сейчас ни квалификация, ни спринт, ни гонка не идут в лайве.")
        lines.append("")
        if next_sess:
            r = next_sess["race"]
            race_name = r.get("raceName") or r.get("circuit", {}).get("circuitName") or "Гран-при"
            label = next_sess["label"]
            start = next_sess["start"]
            cd = format_countdown(start)
            lines.append("Ближайшее важное событие:")
            lines.append(f"{label} {race_name}")
            lines.append(f"Старт: {format_dt_msk(start)} МСК")
            lines.append(f"До старта: {cd}")
            lines.append("")
        else:
            lines.append("В календаре не найдено ближайших квалификаций или гонок.")
            lines.append("")

    lines.append("Очки за ставки:")
    lines.append("• Q1/Q2 (5 вылетевших) — 1 очко за попадание")
    lines.append("• Q3 топ-3: 1 очко за пилота в топ-3, 2 очка при точном месте")
    lines.append("• Спринт топ-3: 2 очка за пилота в топ-3, 3 — за точное место")
    lines.append("• Гран-при топ-3: 3 очка за пилота в топ-3, 5 — за точное место")
    lines.append("• Если угадан весь набор (все вылеты / все места), очки за эту ставку ×2")
    lines.append("")

    def make_bet_button(label_text: str, bet_type: str, is_open: bool):
        if is_open:
            return InlineKeyboardButton(label_text, callback_data=f"bet_menu:{bet_type}")
        else:
            return InlineKeyboardButton(f"{label_text} (окно закрыто)", callback_data="noop")

    keyboard: list[list[InlineKeyboardButton]] = [
        [
            make_bet_button("Q1 — 5 вылетевших", "Q1", q1_open),
        ],
        [
            make_bet_button("Q2 — 5 вылетевших", "Q2", q2_open),
            make_bet_button("Q3 — топ-3", "Q3", q3_open),
        ],
        [
            make_bet_button("Спринт — топ-3", "SPRINT", sprint_open),
            make_bet_button("Гран-при — топ-3", "RACE", race_open),
        ],
        [InlineKeyboardButton("📊 Моя статистика", callback_data="bet_stats:me")],
        [InlineKeyboardButton("📊 Таблица чата", callback_data="bet_stats:chat")],
        [InlineKeyboardButton("⬅️ В главное меню", callback_data="back_main")],
        [
            InlineKeyboardButton("🏁 Итоги Q1", callback_data="bet_settle:Q1"),
            InlineKeyboardButton("🏁 Итоги Q2", callback_data="bet_settle:Q2"),
        ],
        [
            InlineKeyboardButton("🏁 Итоги Q3", callback_data="bet_settle:Q3"),
        ],
        [
            InlineKeyboardButton("🏁 Итоги спринта", callback_data="bet_settle:SPRINT"),
            InlineKeyboardButton("🏁 Итоги гонки", callback_data="bet_settle:RACE"),
        ],
    ]

    await query.edit_message_text(
        "\n".join(lines),
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


# ========= ГЛАВНОЕ МЕНЮ =========

def main_menu_text() -> str:
    return (
        "🏎 F1 Friend League — главное меню\n\n"
        "Что хочешь сделать сейчас?\n\n"
        "📊 Результаты по этапам — посмотреть итоги практик, квалификаций и гонок прошедших Гран-при.\n"
        "🏆 Чемпионаты — открыть таблицу пилотов и конструкторов сезона.\n"
        "📅 Календарь — узнать расписание ближайших гоночных уик-эндов.\n"
        "🎯 Ставки — сделать прогноз на квалификацию и гонку, зарабатывать очки и соревноваться с друзьями.\n\n"
        "Выбери раздел ниже 👇"
    )


def main_menu_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton("📊 Результаты по этапам", callback_data="menu_results")],
        [InlineKeyboardButton("🏆 Чемпионаты", callback_data="menu_standings")],
        [InlineKeyboardButton("📅 Календарь", callback_data="menu_calendar")],
        [InlineKeyboardButton("🎯 Ставки", callback_data="menu_bets")],
    ]
    return InlineKeyboardMarkup(buttons)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    league = get_or_create_league(chat_id)
    year = league["year"]

    text = (
        main_menu_text()
        + f"\n\nТекущий сезон по умолчанию: {year}\n"
        "Можно поменять командой /setyear YYYY."
    )

    if update.message:
        await update.message.reply_text(text, reply_markup=main_menu_keyboard())


# ========= БЛОК "РЕЗУЛЬТАТЫ ПО ЭТАПАМ" =========

RESULT_YEARS = [2025, 2024, 2023]


async def results_entry(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    """Показать выбор сезона для результатов этапов."""
    if query is None and update.callback_query:
        query = update.callback_query
    text = "📊 Выбери сезон, результаты которого хочешь посмотреть:"
    keyboard = [[InlineKeyboardButton(str(y), callback_data=f"res_year:{y}")]
                for y in RESULT_YEARS]
    keyboard.append([InlineKeyboardButton("⬅️ Назад в меню", callback_data="back_main")])

    if query:
        await query.answer()
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
    elif update.message:
        await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))


# ========= CALLBACK-HANDLER ДЛЯ ВСЕХ КНОПОК =========

async def handle_results_callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data or ""
    chat_id = query.message.chat_id
    league = get_chat_league(chat_id)

    # ===== ЧЕМПИОНАТЫ =====

    if data == "menu_standings":
        text = "🏆 Выбери сезон, для которого показать таблицу чемпионата:"
        keyboard = [[InlineKeyboardButton(str(y), callback_data=f"stand_year:{y}")]
                    for y in RESULT_YEARS]
        keyboard.append([InlineKeyboardButton("🏠 В главное меню", callback_data="back_main")])
        await query.answer()
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
        return

    if data.startswith("stand_year:"):
        _, year_str = data.split(":", 1)
        try:
            year = int(year_str)
        except ValueError:
            await query.answer("Некорректный год")
            return

        text = f"🏆 Сезон {year}\nЧто показать?"
        keyboard = [
            [InlineKeyboardButton("👤 Личный зачёт", callback_data=f"stand_drivers:{year}")],
            [InlineKeyboardButton("🏭 Кубок конструкторов", callback_data=f"stand_teams:{year}")],
            [InlineKeyboardButton("⬅️ Выбрать другой сезон", callback_data="menu_standings")],
            [InlineKeyboardButton("🏠 В главное меню", callback_data="back_main")],
        ]
        await query.answer()
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
        return

    if data.startswith("stand_drivers:"):
        _, year_str = data.split(":", 1)
        try:
            year = int(year_str)
        except ValueError:
            await query.answer("Некорректный год")
            return

        try:
            driver_stats, team_stats = compute_season_standings(year)
        except RuntimeError as e:
            await query.answer()
            await query.edit_message_text(str(e), reply_markup=main_menu_keyboard())
            return

        if not driver_stats:
            await query.answer()
            await query.edit_message_text(
                f"За сезон {year} не удалось собрать данные по очкам.",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("⬅️ Назад", callback_data="menu_standings")]]
                ),
            )
            return

        drivers_sorted = sorted(
            driver_stats.values(),
            key=lambda d: (-d["points"], -d.get("wins", 0), d["full_name"]),
        )

        lines = []
        lines.append(f"👤 Личный зачёт — сезон {year}")
        lines.append("")
        header = f"{'POS':<4} {'ACR':<3} {'#':<4} {'DRIVER':<20} {'TEAM':<18} {'PTS':>5} {'WIN':>4}"
        lines.append(header)
        lines.append("-" * len(header))

        pos = 1
        for d in drivers_sorted:
            pos_str = f"{pos}"
            ac = d["acronym"]
            num = d["driver_number"]
            num_str = f"#{num}" if num is not None else "#?"
            name = d["full_name"][:20]
            team = (d.get("last_team") or "")[:18]
            pts_val = d["points"]
            pts = int(pts_val) if abs(pts_val - int(pts_val)) < 0.001 else pts_val
            wins = d.get("wins", 0)
            line = f"{pos_str:<4} {ac:<3} {num_str:<4} {name:<20} {team:<18} {pts:>5} {wins:>4}"
            lines.append(line)
            pos += 1

        keyboard = [
            [InlineKeyboardButton("🏭 Кубок конструкторов", callback_data=f"stand_teams:{year}")],
            [InlineKeyboardButton("⬅️ Выбрать другой сезон", callback_data="menu_standings")],
            [InlineKeyboardButton("🏠 В главное меню", callback_data="back_main")],
        ]

        await query.answer()
        await query.edit_message_text(
            pre_block(lines),
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="HTML",
        )
        return

    if data.startswith("stand_teams:"):
        _, year_str = data.split(":", 1)
        try:
            year = int(year_str)
        except ValueError:
            await query.answer("Некорректный год")
            return

        try:
            driver_stats, team_stats = compute_season_standings(year)
        except RuntimeError as e:
            await query.answer()
            await query.edit_message_text(str(e), reply_markup=main_menu_keyboard())
            return

        if not team_stats:
            await query.answer()
            await query.edit_message_text(
                f"За сезон {year} не удалось собрать очки конструкторов.",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("⬅️ Назад", callback_data="menu_standings")]]
                ),
            )
            return

        teams_sorted = sorted(
            team_stats.values(),
            key=lambda t: (-t["points"], t["team_name"]),
        )

        lines = []
        lines.append(f"🏭 Кубок конструкторов — сезон {year}")
        lines.append("")
        header = f"{'POS':<4} {'TEAM':<24} {'PTS':>5}"
        lines.append(header)
        lines.append("-" * len(header))

        pos = 1
        for t in teams_sorted:
            pos_str = f"{pos}"
            team_name = (t["team_name"] or "")[:24]
            pts_val = t["points"]
            pts = int(pts_val) if abs(pts_val - int(pts_val)) < 0.001 else pts_val
            line = f"{pos_str:<4} {team_name:<24} {pts:>5}"
            lines.append(line)
            pos += 1

        keyboard = [
            [InlineKeyboardButton("👤 Личный зачёт", callback_data=f"stand_drivers:{year}")],
            [InlineKeyboardButton("⬅️ Выбрать другой сезон", callback_data="menu_standings")],
            [InlineKeyboardButton("🏠 В главное меню", callback_data="back_main")],
        ]

        await query.answer()
        await query.edit_message_text(
            pre_block(lines),
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="HTML",
        )
        return

    # ===== РЕЗУЛЬТАТЫ ПО ЭТАПАМ =====

    if data == "menu_results":
        await results_entry(update, context, query=query)
        return

    if data.startswith("res_year:"):
        _, year_str = data.split(":", 1)
        try:
            year = int(year_str)
        except ValueError:
            await query.answer("Некорректный год")
            return

        try:
            meetings = openf1_get("/meetings", {"year": year})
        except Exception:
            logger.exception("meetings error")
            await query.answer("Ошибка загрузки этапов")
            return

        if not meetings:
            await query.answer()
            await query.edit_message_text(
                f"За сезон {year} не найдено этапов.",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("⬅️ Назад", callback_data="menu_results")]]
                ),
            )
            return

        meetings.sort(key=lambda m: m.get("date_start") or "")
        keyboard = []
        for m in meetings:
            mk = m["meeting_key"]
            name = m["meeting_name"]
            keyboard.append(
                [InlineKeyboardButton(name, callback_data=f"res_meeting:{year}:{mk}")]
            )

        keyboard.append(
            [InlineKeyboardButton("⬅️ Назад по сезонам", callback_data="menu_results")]
        )

        await query.answer()
        await query.edit_message_text(
            f"Сезон {year}. Выбери Гран-при:",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
        return

    if data.startswith("res_meeting:"):
        parts = data.split(":")
        _, year_str, mk_str = parts[0], parts[1], parts[2]
        try:
            year = int(year_str)
            meeting_key = int(mk_str)
        except ValueError:
            await query.answer("Некорректные данные")
            return

        try:
            sessions = openf1_get("/sessions", {"meeting_key": meeting_key})
            meetings = openf1_get("/meetings", {"year": year})
        except Exception:
            logger.exception("sessions error")
            await query.answer("Ошибка загрузки сессий")
            return

        meeting = next((m for m in meetings if m["meeting_key"] == meeting_key), None)
        meeting_name = meeting["meeting_name"] if meeting else f"meeting_key={meeting_key}"

        def lower(v):
            return str(v or "").lower()

        session_buttons = []
        for s in sessions:
            stype = lower(s.get("session_type"))
            sname = s.get("session_name") or s.get("session_type") or "Session"
            sid = s["session_key"]

            label = sname
            if "practice" in stype:
                label = f"Practice: {sname}"
            elif "qualifying" in stype:
                label = f"Qualifying: {sname}"
            elif "sprint" in stype:
                label = f"Sprint: {sname}"
            elif "race" in stype:
                label = f"Race: {sname}"

            session_buttons.append(
                [InlineKeyboardButton(label, callback_data=f"res_session:{year}:{meeting_key}:{sid}")]
            )

        if not session_buttons:
            session_buttons.append([InlineKeyboardButton("Нет сессий", callback_data="noop")])

        session_buttons.append(
            [InlineKeyboardButton("⬅️ Назад к Гран-при", callback_data=f"res_year:{year}")]
        )

        await query.answer()
        await query.edit_message_text(
            f"🏁 {meeting_name}\nВыбери сессию:",
            reply_markup=InlineKeyboardMarkup(session_buttons),
        )
        return

    if data.startswith("res_session:"):
        _, year_str, mk_str, sid_str = data.split(":", 3)
        try:
            year = int(year_str)
            meeting_key = int(mk_str)
            session_key = int(sid_str)
        except ValueError:
            await query.answer("Некорректные данные")
            return

        try:
            meetings = openf1_get("/meetings", {"year": year})
            sessions = openf1_get("/sessions", {"meeting_key": meeting_key})
            drivers_raw = openf1_get("/drivers", {"meeting_key": meeting_key})
            results = openf1_get("/session_result", {"session_key": session_key})
        except Exception:
            logger.exception("session result error")
            await query.answer("Ошибка загрузки результатов")
            return

        meeting = next((m for m in meetings if m["meeting_key"] == meeting_key), None)
        meeting_name = meeting["meeting_name"] if meeting else f"meeting_key={meeting_key}"

        session = next((s for s in sessions if s["session_key"] == session_key), None)
        session_name = (
            session.get("session_name") or session.get("session_type") or "Session"
            if session else "Session"
        )
        session_type = (session.get("session_type") or "").lower() if session else ""

        drivers = dedupe_drivers(drivers_raw)
        driver_by_num = {d["driver_number"]: d for d in drivers}

        sorted_res = sort_results_by_position(results)

        lines = []
        lines.append(f"{meeting_name}")
        lines.append(f"Сессия: {session_name}")
        lines.append("")
        lines.append("'>  — обладатель быстрейшего круга")
        lines.append("")

        if not sorted_res:
            lines.append("Нет результатов по этой сессии.")
        else:
            leader_duration = None
            for r in sorted_res:
                dur = extract_last_segment(r.get("duration"))
                if dur not in (None, 0, "0", "0.0", "0.000"):
                    leader_duration = dur
                    break

            best_lap_by_driver: dict[int, float] = {}
            fastest_best_lap: float | None = None
            fastest_driver_num: int | None = None

            if "race" in session_type:
                try:
                    laps = openf1_get("/laps", {"session_key": session_key})
                except Exception:
                    laps = []

                for lap in laps:
                    num = lap.get("driver_number")
                    dur = lap.get("lap_duration")
                    if num is None or dur is None:
                        continue
                    if lap.get("is_pit_out_lap"):
                        continue
                    try:
                        dur_f = float(dur)
                    except (TypeError, ValueError):
                        continue
                    if num not in best_lap_by_driver or dur_f < best_lap_by_driver[num]:
                        best_lap_by_driver[num] = dur_f

                for num, dur_f in best_lap_by_driver.items():
                    if fastest_best_lap is None or dur_f < fastest_best_lap:
                        fastest_best_lap = dur_f
                        fastest_driver_num = num
            else:
                for r in sorted_res:
                    num = r.get("driver_number")
                    if num is None:
                        continue
                    dur = r.get("duration")
                    best = None
                    if isinstance(dur, list):
                        vals = [v for v in dur if v not in (None, 0, "0", "0.0", "0.000")]
                        if not vals:
                            continue
                        try:
                            best = min(float(v) for v in vals)
                        except Exception:
                            continue
                    elif dur not in (None, 0, "0.0", "0.000", 0):
                        try:
                            best = float(dur)
                        except (TypeError, ValueError):
                            continue

                    if best is None:
                        continue

                    best_lap_by_driver[num] = best
                    if fastest_best_lap is None or best < fastest_best_lap:
                        fastest_best_lap = best
                        fastest_driver_num = num

            header = (
                f"{'':1}{'POS':<4} {'ACR':<3} {'#':<4} "
                f"{'DRIVER':<20} {'TEAM':<18} {'STAT':<5} {'TIME/GAP':<12} {'BEST LAP':<12}"
            )
            lines.append(header)
            lines.append("-" * len(header))

            for r in sorted_res:
                pos = r.get("position")
                num = r.get("driver_number")
                d = driver_by_num.get(num)

                if isinstance(pos, int):
                    pos_str = f"P{pos}"
                else:
                    pos_str = "P?"

                if d:
                    ac = (d.get("name_acronym") or "").upper()
                    name = d["full_name"]
                    team = d.get("team_name") or ""
                    num_val = d.get("driver_number") or num
                else:
                    ac = "???"
                    name = "Unknown"
                    team = ""
                    num_val = num

                num_str = f"#{num_val}" if num_val is not None else "#?"
                name_col = name[:20]
                team_col = team[:18]

                status_label = ""
                if r.get("dsq"):
                    status_label = "DSQ"
                elif r.get("dns"):
                    status_label = "DNS"
                elif r.get("dnf"):
                    status_label = "DNF"

                time_or_gap = ""

                if isinstance(pos, int) and pos == 1:
                    dur_val = extract_last_segment(r.get("duration"))
                    if dur_val is None and leader_duration is not None:
                        dur_val = leader_duration
                    if dur_val is not None:
                        time_or_gap = format_seconds_like_laptime(dur_val)
                else:
                    gap_val = extract_last_segment(r.get("gap_to_leader"))
                    if gap_val is not None:
                        if isinstance(gap_val, (int, float)) or (
                            isinstance(gap_val, str)
                            and gap_val.replace(".", "", 1).isdigit()
                        ):
                            formatted_gap = format_seconds_like_laptime(gap_val)
                            time_or_gap = f"+{formatted_gap}"
                        else:
                            time_or_gap = str(gap_val)

                best_lap_col = ""
                if (
                    fastest_driver_num is not None
                    and num_val is not None
                    and num_val == fastest_driver_num
                ):
                    val = best_lap_by_driver.get(num_val)
                    if val is not None:
                        best_lap_col = format_seconds_like_laptime(val)

                mark = ">"
                if not (
                    fastest_driver_num is not None
                    and num_val is not None
                    and num_val == fastest_driver_num
                ):
                    mark = " "

                line = (
                    f"{mark:1}{pos_str:<4} {ac:<3} {num_str:<4} "
                    f"{name_col:<20} {team_col:<18} {status_label:<5} {time_or_gap:<12} {best_lap_col:<12}"
                )
                lines.append(line)

            if fastest_driver_num is not None and fastest_best_lap is not None:
                d_fl = driver_by_num.get(fastest_driver_num)
                if d_fl:
                    ac_fl = (d_fl.get("name_acronym") or "").upper()
                    name_fl = d_fl["full_name"]
                    num_fl = d_fl.get("driver_number") or fastest_driver_num
                    fl_time_str = format_seconds_like_laptime(fastest_best_lap)
                    lines.append("")
                    lines.append(
                        f"Fastest lap: {fl_time_str} — {ac_fl} #{num_fl} {name_fl}"
                    )

        keyboard = [
            [InlineKeyboardButton("⬅️ Назад к сессиям", callback_data=f"res_meeting:{year}:{meeting_key}")],
            [InlineKeyboardButton("⬅️ Назад к Гран-при", callback_data=f"res_year:{year}")],
            [InlineKeyboardButton("🏠 В главное меню", callback_data="back_main")],
        ]

        await query.answer()
        await query.edit_message_text(
            pre_block(lines),
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="HTML",
        )
        return

    # ===== КАЛЕНДАРЬ =====

    if data == "menu_calendar":
        today_msk = datetime.now(MSK_TZ).date()
        year = today_msk.year

        try:
            races = load_f1api_races(year)
        except Exception as e:
            logger.exception("calendar load error")
            await query.answer()
            await query.edit_message_text(
                f"Не удалось загрузить календарь сезона {year}:\n{e}",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("🏠 В главное меню", callback_data="back_main")]]
                ),
            )
            return

        future_races = []
        past_races = []

        for r in races:
            sch = r.get("schedule") or {}
            race_sch = sch.get("race") or {}
            date_str = race_sch.get("date")
            race_date = None
            if date_str:
                try:
                    race_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                except Exception:
                    race_date = None

            if race_date and race_date >= today_msk:
                future_races.append(r)
            else:
                past_races.append(r)

        future_races.sort(key=lambda rr: rr.get("round") or 999)
        past_races.sort(key=lambda rr: rr.get("round") or 999)

        lines = []
        lines.append(f"📅 Календарь сезона {year}")
        lines.append("")
        lines.append("Выбери этап, чтобы увидеть расписание уик-энда в МСК:")
        lines.append("")

        keyboard: list[list[InlineKeyboardButton]] = []

        def make_label(r: dict) -> tuple[str, str]:
            race_id = r.get("raceId") or f"{r.get('season', year)}_{r.get('round')}"
            round_no = r.get("round") or 0
            round_str = f"R{int(round_no):02d}" if isinstance(round_no, int) else "R??"

            sch = r.get("schedule") or {}
            race_sch = sch.get("race") or {}
            d = race_sch.get("date") or "????-??-??"

            race_name = r.get("raceName")
            circuit = r.get("circuit") or {}
            country = circuit.get("country") or ""
            name = race_name or circuit.get("circuitName") or f"Round {round_no or '?'}"
            if country:
                main = f"{name} ({country})"
            else:
                main = name

            label = f"{round_str} • {d} • {main}"
            return race_id, label

        # СНАЧАЛА ПРОШЕДШИЕ ЭТАПЫ
        for r in past_races:
            race_id, label = make_label(r)
            keyboard.append(
                [InlineKeyboardButton(label, callback_data=f"cal_f1api:{race_id}")]
            )

        # Разделитель + БУДУЩИЕ ЭТАПЫ ВНИЗУ
        if future_races:
            keyboard.append([InlineKeyboardButton("— Будущие этапы —", callback_data="noop")])
            for r in future_races:
                race_id, label = make_label(r)
                keyboard.append(
                    [InlineKeyboardButton(label, callback_data=f"cal_f1api:{race_id}")]
                )

        keyboard.append(
            [InlineKeyboardButton("🏠 В главное меню", callback_data="back_main")]
        )

        await query.answer()
        await query.edit_message_text(
            "\n".join(lines),
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
        return

    if data.startswith("cal_f1api:"):
        _, race_id = data.split(":", 1)

        race = F1API_RACES_BY_ID.get(race_id)

        if not race:
            year = None
            suffix = race_id[-4:]
            if suffix.isdigit():
                year = int(suffix)
            if year is None:
                year = datetime.now(MSK_TZ).year
            try:
                load_f1api_races(year)
            except Exception:
                pass
            race = F1API_RACES_BY_ID.get(race_id)

        if not race:
            await query.answer("Этап не найден.")
            return

        season = race.get("season")
        rnd = race.get("round")
        race_name = race.get("raceName")
        circuit = race.get("circuit") or {}
        country = circuit.get("country") or ""
        city = circuit.get("city") or ""
        circuit_name = circuit.get("circuitName") or ""

        sch = race.get("schedule") or {}

        def fmt_session(label: str, key: str) -> str | None:
            s = sch.get(key) or {}
            d = s.get("date")
            t = s.get("time")
            if not d and not t:
                return None
            dt = parse_utc_to_msk_dt(d, t)
            if dt:
                local = format_dt_msk(dt)
                return f"{label}: {local} (МСК)"
            else:
                raw = f"{d or ''} {t or ''}".strip()
                return f"{label}: {raw}" if raw else None

        def fmt_sessions_block(sch_dict: dict) -> list[str]:
            out = []
            for label, key in [
                ("FP1", "fp1"),
                ("FP2", "fp2"),
                ("FP3", "fp3"),
                ("Sprint Qualifying", "sprintQualy"),
                ("Sprint Race", "sprintRace"),
                ("Qualifying", "qualy"),
                ("Race", "race"),
            ]:
                row = fmt_session(label, key)
                if row:
                    out.append(row)
            return out

        lines = []
        title = race_name or circuit_name or f"Round {rnd}"
        lines.append(f"📅 {title}")
        place = ", ".join(x for x in [circuit_name, city, country] if x)
        if place:
            lines.append(place)
        if season and rnd:
            lines.append(f"Сезон {season}, этап {rnd}")
        lines.append("")
        lines.append("Расписание уик-энда (МСК):")
        lines.append("")

        session_lines = fmt_sessions_block(sch)
        if session_lines:
            lines.extend(session_lines)
        else:
            lines.append("Нет подробного расписания по этому этапу.")

        keyboard = [
            [InlineKeyboardButton("⬅️ Назад к календарю", callback_data="menu_calendar")],
            [InlineKeyboardButton("🏠 В главное меню", callback_data="back_main")],
        ]

        await query.answer()
        await query.edit_message_text(
            "\n".join(lines),
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
        return

    # ===== СТАВКИ (Q1, Q2, Q3, SPRINT, RACE) =====

    if data == "menu_bets":
        await query.answer()
        await open_bets_menu(query, league)
        return

    if data.startswith("bet_menu:"):
        _, bet_type = data.split(":", 1)

        if bet_type not in BET_MAX:
            await query.answer("Неизвестный тип ставки.")
            return

        # проверяем окно ставок (5 минут)
        if not is_bet_window_open(bet_type, league):
            await query.answer(
                "Окно ставок для этого типа сейчас закрыто.\n"
                "Ставки принимаются только в первые 5 минут соответствующей сессии.",
                show_alert=True,
            )
            await open_bets_menu(query, league)
            return

        drivers = league.get("drivers") or []
        meeting = league.get("meeting")
        if not drivers or not meeting:
            await query.answer("Сейчас попробую подтянуть текущий Гран-при...")
            await open_bets_menu(query, league)
            return

        meeting_key = meeting["meeting_key"]

        # базовый список — все пилоты
        allowed_nums = sorted(
            d["driver_number"]
            for d in drivers
            if d.get("driver_number") is not None
        )

        # если есть данные квалификации — пытаемся сузить Q2/Q3
        if bet_type in ("Q2", "Q3") and league.get("qual_results"):
            q1_out, q2_out, q3_top = split_qual_results(league["qual_results"])
            q1_nums = {r["driver_number"] for r in q1_out}
            q2_nums = {r["driver_number"] for r in q2_out}
            if bet_type == "Q2":
                pass
            elif bet_type == "Q3":
                allowed_nums = [n for n in allowed_nums if n not in q1_nums and n not in q2_nums]

        league.setdefault("pending_bets", {})
        league["pending_bets"][query.from_user.id] = {
            "type": bet_type,
            "meeting_key": meeting_key,
            "allowed_drivers": allowed_nums,
            "selected": [],
            "max_count": BET_MAX[bet_type],
        }

        kb = build_bet_keyboard(bet_type, league, query.from_user.id)
        text_map = {
            "Q1": "Q1: выбери до 5 пилотов, которые вылетят в Q1.",
            "Q2": "Q2: выбери до 5 пилотов, которые вылетят в Q2.",
            "Q3": "Q3: выбери 1–3 пилотов, которые займут 1–3 места в квалификации.",
            "SPRINT": "Спринт: выбери 1–3 пилотов, которые займут топ-3 в спринт-гонке.",
            "RACE": "Гран-при: выбери 1–3 пилотов, которые займут топ-3 в гонке.",
        }
        text = text_map.get(bet_type, "Выбери пилотов для ставки.")
        await query.answer()
        await query.edit_message_text(text, reply_markup=kb)
        return

    if data.startswith("bet_pick:"):
        _, bet_type, num_str = data.split(":", 2)
        pending = league.setdefault("pending_bets", {})
        state = pending.get(query.from_user.id)
        if not state or state.get("type") != bet_type:
            await query.answer("Нет активного выбора для этого типа ставки.")
            return

        try:
            num = int(num_str)
        except ValueError:
            await query.answer()
            return

        if num not in state["allowed_drivers"]:
            await query.answer("Этот пилот недоступен для выбора.")
            return

        sel = state["selected"]
        if num in sel:
            sel.remove(num)
        else:
            if len(sel) >= state["max_count"]:
                await query.answer(f"Можно выбрать максимум {state['max_count']} пилотов.")
                return
            sel.append(num)

        kb = build_bet_keyboard(bet_type, league, query.from_user.id)
        await query.answer()
        await query.edit_message_reply_markup(reply_markup=kb)
        return

    if data.startswith("bet_confirm:"):
        _, bet_type = data.split(":", 1)
        pending = league.setdefault("pending_bets", {})
        state = pending.get(query.from_user.id)
        if not state or state.get("type") != bet_type:
            await query.answer("Нет активного выбора для этого типа ставки.")
            return

        # снова проверяем окно ставок
        if not is_bet_window_open(bet_type, league):
            await query.answer(
                "К моменту подтверждения окно ставок уже закрыто.\n"
                "Ставка не принята.",
                show_alert=True,
            )
            pending.pop(query.from_user.id, None)
            await open_bets_menu(query, league)
            return

        selected = state["selected"]
        if not selected:
            await query.answer("Ты ещё никого не выбрал.", show_alert=True)
            return

        meeting_key = state["meeting_key"]
        league.setdefault("bets", {}).setdefault(meeting_key, {}).setdefault(
            bet_type, {"bets": {}, "settled": False}
        )
        league["bets"][meeting_key][bet_type]["bets"][query.from_user.id] = list(selected)

        # очищаем состояние выбора
        pending.pop(query.from_user.id, None)

        await query.answer()
        await query.edit_message_text(
            f"✅ Ставка на {bet_type} сохранена!\n"
            f"Выбранные машины: {', '.join(str(n) for n in selected)}"
        )
        return

    if data.startswith("bet_cancel:"):
        _, bet_type = data.split(":", 1)
        pending = league.setdefault("pending_bets", {})
        pending.pop(query.from_user.id, None)
        await query.answer(f"Выбор для {bet_type} отменён.")
        await open_bets_menu(query, league)
        return

    if data == "bet_stats:me":
        xp = league.get("xp", {})
        points = xp.get(query.from_user.id, 0)
        await query.answer()
        await query.edit_message_text(
            f"📊 Твоя статистика:\n\n"
            f"Очки (опыт) за все Гран-при в этом чате: {points}",
            reply_markup=InlineKeyboardMarkup(
                [[InlineKeyboardButton("⬅️ Назад к ставкам", callback_data="menu_bets")]]
            ),
        )
        return

    if data == "bet_stats:chat":
        xp = league.get("xp", {})
        if not xp:
            await query.answer()
            await query.edit_message_text(
                "В этом чате пока нет ни одного начисления очков.",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("⬅️ Назад к ставкам", callback_data="menu_bets")]]
                ),
            )
            return

        rows = sorted(xp.items(), key=lambda kv: kv[1], reverse=True)
        lines = ["📊 Таблица чата (по очкам ставок):", ""]
        pos = 1
        for user_id, pts in rows:
            try:
                user_chat = await context.bot.get_chat(user_id)
                name = user_chat.first_name or str(user_id)
            except Exception:
                name = str(user_id)
            lines.append(f"{pos}. {name} — {pts}")
            pos += 1

        await query.answer()
        await query.edit_message_text(
            "\n".join(lines),
            reply_markup=InlineKeyboardMarkup(
                [[InlineKeyboardButton("⬅️ Назад к ставкам", callback_data="menu_bets")]]
            ),
        )
        return

    # ===== ПОДВЕДЕНИЕ ИТОГОВ СТАВОК =====

    if data.startswith("bet_settle:"):
        _, bet_type = data.split(":", 1)
        meeting = league.get("meeting")
        if not meeting:
            await query.answer()
            await query.edit_message_text(
                "Сначала нужно выбрать текущий Гран-при и загрузить его данные (зайди в ставки ещё раз).",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("⬅️ Назад к ставкам", callback_data="menu_bets")]]
                ),
            )
            return

        meeting_key = meeting["meeting_key"]
        bets_meeting = league.setdefault("bets", {}).setdefault(meeting_key, {})
        block = bets_meeting.setdefault(bet_type, {"bets": {}, "settled": False})
        bets = block.get("bets") or {}

        if block.get("settled"):
            await query.answer()
            await query.edit_message_text(
                f"Итоги для {bet_type} уже были подведены для этого Гран-при.",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("⬅️ Назад к ставкам", callback_data="menu_bets")]]
                ),
            )
            return

        if not bets:
            await query.answer()
            await query.edit_message_text(
                f"Никто ещё не сделал ставку на {bet_type} для этого Гран-при.",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("⬅️ Назад к ставкам", callback_data="menu_bets")]]
                ),
            )
            return

        # общие данные
        drivers = league.get("drivers") or []
        driver_by_num = {d["driver_number"]: d for d in drivers}
        xp = league.setdefault("xp", {})

        # ====== Q1 / Q2 (вылеты) ======
        if bet_type in ("Q1", "Q2"):
            qual_results = league.get("qual_results") or []
            if not qual_results:
                await query.answer()
                await query.edit_message_text(
                    "Нет данных квалификации для этого Гран-при.\n"
                    "Попробуй позже, когда они появятся.",
                    reply_markup=InlineKeyboardMarkup(
                        [[InlineKeyboardButton("⬅️ Назад к ставкам", callback_data="menu_bets")]]
                    ),
                )
                return

            q1_out, q2_out, q3_top = split_qual_results(qual_results)
            if bet_type == "Q1":
                real_out_list = q1_out
                title = "Q1"
            else:
                real_out_list = q2_out
                title = "Q2"

            real_nums = {r["driver_number"] for r in real_out_list}
            per_hit = BET_POINTS_ELIM[bet_type]

            lines = []
            lines.append(f"🏁 Итоги ставок {title}")
            lines.append("")
            lines.append(f"Фактический {title} OUT:")
            for r in real_out_list:
                d = driver_by_num.get(r["driver_number"])
                ac = (d.get("name_acronym") or "").upper() if d else "???"
                name = d["full_name"] if d else "Unknown"
                pos = r.get("position")
                pos_str = f"P{pos}" if isinstance(pos, int) else "P?"
                lines.append(f"{pos_str}: {ac} — {name}")
            lines.append("")
            lines.append(f"Очки за {title} (1 попадание = 1 очко, полный угадыш ×2):")

            hits_table: list[tuple[int, str, int, int]] = []

            for user_id, selected_nums in bets.items():
                selected_set = set(selected_nums)
                hits = len(real_nums.intersection(selected_set))
                pts = hits * per_hit
                if hits == len(real_nums) and len(selected_set) == len(real_nums) and pts > 0:
                    pts *= 2  # множитель за идеальный прогноз

                if pts > 0:
                    xp[user_id] = xp.get(user_id, 0) + pts

                try:
                    user_chat = await context.bot.get_chat(user_id)
                    name = user_chat.first_name or str(user_id)
                except Exception:
                    name = str(user_id)
                hits_table.append((user_id, name, hits, pts))

            if not hits_table:
                lines.append("Никто не угадал ни одного пилота 😅")
            else:
                for _, name, hits, pts in hits_table:
                    lines.append(f"{name}: {hits} попаданий (+{pts} очков)")

            if xp:
                lines.append("")
                lines.append("📊 Общая таблица чата:")
                rows = sorted(xp.items(), key=lambda kv: kv[1], reverse=True)
                pos = 1
                for user_id, pts in rows:
                    try:
                        user_chat = await context.bot.get_chat(user_id)
                        name = user_chat.first_name or str(user_id)
                    except Exception:
                        name = str(user_id)
                    lines.append(f"{pos}. {name} — {pts}")
                    pos += 1

            block["settled"] = True

            await query.answer()
            await query.edit_message_text(
                "\n".join(lines),
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("⬅️ Назад к ставкам", callback_data="menu_bets")]]
                ),
            )
            return

        # ====== Q3 / SPRINT / RACE (топ-3) ======
        if bet_type in ("Q3", "SPRINT", "RACE"):

            # определяем, какой session_key брать
            session_key = None
            session_label = ""
            if bet_type == "Q3":
                if league.get("qual_results"):
                    try:
                        sessions = openf1_get("/sessions", {"meeting_key": meeting_key})
                    except Exception:
                        sessions = []
                    def lower(v): return str(v or "").lower()
                    qual_sess = None
                    for s in sessions:
                        stype = lower(s.get("session_type"))
                        if ("qualifying" in stype or "short qualifying" in stype) and "sprint" not in stype:
                            qual_sess = s
                            break
                    if qual_sess:
                        session_key = qual_sess["session_key"]
                session_label = "Q3 (квалификация)"
            elif bet_type == "SPRINT":
                try:
                    sessions = openf1_get("/sessions", {"meeting_key": meeting_key})
                except Exception:
                    sessions = []
                def lower(v): return str(v or "").lower()
                spr_sess = None
                for s in sessions:
                    stype = lower(s.get("session_type"))
                    if "sprint" in stype and "race" in stype:
                        spr_sess = s
                        break
                if spr_sess:
                    session_key = spr_sess["session_key"]
                session_label = "Спринт"
            elif bet_type == "RACE":
                try:
                    sessions = openf1_get("/sessions", {"meeting_key": meeting_key})
                except Exception:
                    sessions = []
                def lower(v): return str(v or "").lower()
                race_sess = None
                for s in sessions:
                    stype = lower(s.get("session_type"))
                    if "race" in stype and "sprint" not in stype:
                        race_sess = s
                        break
                if race_sess:
                    session_key = race_sess["session_key"]
                session_label = "Гонка"

            if not session_key:
                await query.answer()
                await query.edit_message_text(
                    f"Не удалось найти сессию для {bet_type}.\n"
                    "Попробуй позже.",
                    reply_markup=InlineKeyboardMarkup(
                        [[InlineKeyboardButton("⬅️ Назад к ставкам", callback_data="menu_bets")]]
                    ),
                )
                return

            try:
                results = openf1_get("/session_result", {"session_key": session_key})
            except Exception:
                logger.exception("session_result error for %s", bet_type)
                await query.answer()
                await query.edit_message_text(
                    f"Не удалось загрузить результаты {session_label}.\n"
                    "Попробуй позже.",
                    reply_markup=InlineKeyboardMarkup(
                        [[InlineKeyboardButton("⬅️ Назад к ставкам", callback_data="menu_bets")]]
                    ),
                )
                return

            sorted_res = sort_results_by_position(results)
            top3 = [
                r for r in sorted_res
                if isinstance(r.get("position"), int) and 1 <= r["position"] <= 3
            ]
            if len(top3) < 3:
                await query.answer()
                await query.edit_message_text(
                    f"В результатах {session_label} топ-3 ещё не сформирован.\n"
                    "Попробуй позже.",
                    reply_markup=InlineKeyboardMarkup(
                        [[InlineKeyboardButton("⬅️ Назад к ставкам", callback_data="menu_bets")]]
                    ),
                )
                return

            real_nums_ordered = [r["driver_number"] for r in top3]
            real_nums_set = set(real_nums_ordered)

            pts_cfg = TOP3_POINTS[bet_type]
            base_pts = pts_cfg["in_top"]
            exact_pts = pts_cfg["exact"]

            lines = []
            lines.append(f"🏁 Итоги ставок {session_label}")
            lines.append("")
            lines.append(f"Фактический топ-3 {session_label}:")
            for r in top3:
                d = driver_by_num.get(r["driver_number"])
                ac = (d.get("name_acronym") or "").upper() if d else "???"
                name = d["full_name"] if d else "Unknown"
                pos = r.get("position")
                pos_str = f"P{pos}" if isinstance(pos, int) else "P?"
                lines.append(f"{pos_str}: {ac} — {name}")
            lines.append("")
            lines.append(
                f"Очки: пилот в топ-3 = {base_pts}, точное место = {exact_pts}. "
                "Если угадан весь топ-3 по местам — очки ×2."
            )
            lines.append("")

            # считаем каждому
            results_rows: list[tuple[int, str, int, int]] = []

            for user_id, selected_nums in bets.items():
                total_pts = 0
                hits_any = 0

                for i in range(3):
                    if i >= len(selected_nums):
                        continue
                    guess = selected_nums[i]
                    actual = real_nums_ordered[i]
                    if guess == actual:
                        total_pts += exact_pts
                        hits_any += 1
                    elif guess in real_nums_set:
                        total_pts += base_pts
                        hits_any += 1

                perfect = False
                if len(selected_nums) >= 3:
                    if selected_nums[:3] == real_nums_ordered:
                        perfect = True

                if perfect and total_pts > 0:
                    total_pts *= 2

                if total_pts > 0:
                    xp[user_id] = xp.get(user_id, 0) + total_pts

                try:
                    user_chat = await context.bot.get_chat(user_id)
                    name = user_chat.first_name or str(user_id)
                except Exception:
                    name = str(user_id)
                results_rows.append((user_id, name, hits_any, total_pts))

            if not results_rows:
                lines.append("Никто не угадал ни одного пилота в топ-3 😅")
            else:
                for _, name, hits, pts in results_rows:
                    lines.append(f"{name}: {hits} попаданий (+{pts} очков)")
            lines.append("")

            if xp:
                lines.append("📊 Общая таблица чата:")
                rows = sorted(xp.items(), key=lambda kv: kv[1], reverse=True)
                pos = 1
                for user_id, pts in rows:
                    try:
                        user_chat = await context.bot.get_chat(user_id)
                        name = user_chat.first_name or str(user_id)
                    except Exception:
                        name = str(user_id)
                    lines.append(f"{pos}. {name} — {pts}")
                    pos += 1

            block["settled"] = True

            await query.answer()
            await query.edit_message_text(
                "\n".join(lines),
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("⬅️ Назад к ставкам", callback_data="menu_bets")]]
                ),
            )
            return

    # ===== ПРОЧЕЕ И МЕНЮ =====

    if data == "back_main":
        await query.answer()
        await query.edit_message_text(
            main_menu_text(),
            reply_markup=main_menu_keyboard(),
        )
        return

    if data == "noop":
        await query.answer()
        return

    await query.answer()


# ========= КОМАНДЫ /setyear И ОБЩИЙ ЭХО =========

async def setyear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    league = get_or_create_league(chat_id)
    if not context.args:
        if update.message:
            await update.message.reply_text("Использование: /setyear 2025")
        return
    try:
        year = int(context.args[0])
    except ValueError:
        if update.message:
            await update.message.reply_text("Год должен быть числом, например: /setyear 2025")
        return

    league["year"] = year
    league["meeting"] = None
    league["drivers"] = []
    league["qual_results"] = []
    league["phase"] = "IDLE"
    league["bets_q1"] = {}
    league["bets"] = {}
    league["pending_bets"] = {}
    # xp оставляем, как «общий опыт» по чату

    if update.message:
        await update.message.reply_text(
            f"Сезон для результатов и ставок установлен: {year}.\n"
            "Календарь для этого года подтянется автоматически, "
            "а текущий Гран-при для ставок будет выбран как последний завершённый этап сезона."
        )


async def echo_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message and update.message.text and update.message.text.startswith("/"):
        return
    if update.message:
        await update.message.reply_text("Используй /start, чтобы открыть меню.")


# ========= ЗАПУСК ПРИЛОЖЕНИЯ =========

def main():
    if TELEGRAM_TOKEN == "PASTE_YOUR_TOKEN_HERE" or not TELEGRAM_TOKEN:
        raise RuntimeError(
            "Сначала установи токен в TELEGRAM_TOKEN "
            "(в коде или через переменную окружения TELEGRAM_TOKEN)"
        )

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("setyear", setyear))

    app.add_handler(CallbackQueryHandler(handle_results_callbacks))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo_help))

    logger.info("Bot starting...")
    app.run_polling()


if __name__ == "__main__":
    main()
