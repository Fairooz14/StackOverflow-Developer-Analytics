import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── Dataset size ─────────────────────────────────────────────────────────────
N_USERS     = 50_000
N_QUESTIONS = 200_000
N_ANSWERS   = 350_000

TAGS_RAW = [
    ("python",          2_200_000, 2008, "rising"),
    ("javascript",      2_100_000, 2008, "stable"),
    ("java",            1_800_000, 2008, "stable"),
    ("sql",               650_000, 2008, "stable"),
    ("c#",              1_500_000, 2008, "stable"),
    ("php",             1_400_000, 2008, "declining"),
    ("html",            1_100_000, 2008, "stable"),
    ("css",               800_000, 2008, "stable"),
    ("react",             450_000, 2015, "rising"),
    ("node.js",           400_000, 2011, "rising"),
    ("typescript",        380_000, 2016, "rising"),
    ("c++",               780_000, 2008, "stable"),
    ("android",         1_000_000, 2010, "declining"),
    ("mysql",             650_000, 2008, "stable"),
    ("postgresql",        280_000, 2010, "rising"),
    ("pandas",            280_000, 2012, "rising"),
    ("numpy",             200_000, 2012, "stable"),
    ("tensorflow",        160_000, 2017, "rising"),
    ("docker",            200_000, 2015, "rising"),
    ("git",               300_000, 2010, "stable"),
    ("linux",             500_000, 2008, "stable"),
    ("aws",               250_000, 2014, "rising"),
    ("r",                 430_000, 2009, "stable"),
    ("swift",             280_000, 2015, "rising"),
    ("kotlin",            180_000, 2017, "rising"),
    ("go",                150_000, 2015, "rising"),
    ("rust",               90_000, 2018, "rising"),
    ("django",            300_000, 2010, "stable"),
    ("flask",             200_000, 2011, "stable"),
    ("machine-learning",  230_000, 2013, "rising"),
    ("deep-learning",     110_000, 2016, "rising"),
    ("data-science",      140_000, 2014, "rising"),
    ("excel",             500_000, 2008, "stable"),
    ("bash",              300_000, 2008, "stable"),
    ("mongodb",           200_000, 2013, "stable"),
    ("redis",             130_000, 2014, "stable"),
    ("kubernetes",        120_000, 2018, "rising"),
    ("spark",             100_000, 2015, "stable"),
    ("scikit-learn",       90_000, 2014, "rising"),
]

# ── Realistic question title templates per tag ────────────────────────────────
TITLE_TEMPLATES = {
    "python":           ["How to {verb} a list in Python?",
                         "Python {noun} not working as expected",
                         "Best way to {verb} {noun} in Python",
                         "Python {noun} vs {noun2} — which to use?"],
    "javascript":       ["JavaScript {noun} returns undefined",
                         "How to {verb} async/await in JavaScript",
                         "Why does {noun} behave differently in JS?",
                         "Fixing {noun} error in JavaScript"],
    "java":             ["Java {noun} throws NullPointerException",
                         "How to {verb} {noun} in Java 17?",
                         "Java {noun} vs {noun2} performance"],
    "react":            ["React {noun} not re-rendering correctly",
                         "How to {verb} state in React hooks",
                         "React {noun} component best practices",
                         "UseEffect {noun} infinite loop fix"],
    "sql":              ["SQL query to {verb} duplicate {noun}",
                         "Optimising slow {noun} SQL query",
                         "JOIN vs subquery for {noun}",
                         "How to {verb} {noun} in SQL?"],
    "python+pandas":    ["Pandas {noun} filtering not working",
                         "How to {verb} DataFrame rows by condition",
                         "Pandas groupby {noun} aggregation"],
    "docker":           ["Docker container {noun} not starting",
                         "How to {verb} volumes in Docker Compose",
                         "Docker networking: {noun} unreachable"],
    "kubernetes":       ["Kubernetes pod stuck in {noun} state",
                         "How to {verb} secrets in K8s",
                         "K8s {noun} deployment keeps crashing"],
    "machine-learning": ["How to {verb} overfitting in {noun} model",
                         "Choosing the right {noun} metric",
                         "ML model {noun} accuracy too low"],
    "_default":         ["How to {verb} {noun} efficiently?",
                         "{noun} not working — need help",
                         "Best practice for {verb}ing {noun}",
                         "Difference between {noun} and {noun2}",
                         "Getting {noun} error — how to fix?",
                         "How do I {verb} {noun} without {noun2}?"],
}

_VERBS  = ["parse", "filter", "sort", "merge", "transform", "serialize",
           "validate", "optimise", "debug", "handle", "format", "iterate",
           "convert", "cache", "encrypt", "deploy", "test", "mock", "map",
           "flatten", "aggregate", "paginate", "authenticate", "inject"]
_NOUNS  = ["dictionary", "list", "class", "function", "object", "variable",
           "string", "integer", "DataFrame", "query", "response", "request",
           "exception", "thread", "process", "module", "package", "session",
           "token", "config", "schema", "pipeline", "callback", "promise",
           "stream", "socket", "hook", "middleware", "endpoint", "payload"]
_NOUNS2 = ["array", "tuple", "map", "set", "generator", "iterator", "struct",
           "interface", "type", "enum", "record", "union", "context", "state"]

# ── Locations with realistic weights ─────────────────────────────────────────
LOCATIONS = [
    ("United States", 0.26), ("India", 0.20), ("Germany", 0.07),
    ("United Kingdom", 0.07), ("Canada", 0.06), ("France", 0.05),
    ("Brazil", 0.05), ("Bangladesh", 0.03), ("Australia", 0.04),
    ("Netherlands", 0.02), ("Poland", 0.02), ("Russia", 0.02),
    ("China", 0.02), ("Japan", 0.02), ("Pakistan", 0.01),
    ("Spain", 0.01), ("Italy", 0.01), ("South Korea", 0.01),
    ("Sweden", 0.01), ("", 0.06),
]

# ── Badge definitions ─────────────────────────────────────────────────────────
GOLD_BADGES   = ["Famous Question", "Good Answer", "Great Question",
                 "Legendary", "Populist", "Guru"]
SILVER_BADGES = ["Civic Duty", "Enlightened", "Nice Answer",
                 "Nice Question", "Pundit", "Revival"]
BRONZE_BADGES = ["Altruist", "Analytical", "Autobiographer",
                 "Commentator", "Critic", "Editor", "Organiser",
                 "Scholar", "Student", "Supporter", "Teacher"]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tag_weight_at_year(tag_name, launch_year, trajectory, year):
    """Return a multiplier for a tag's popularity in a given year."""
    if year < launch_year:
        return 0.0
    age = year - launch_year
    if trajectory == "rising":
        return min(1.0, 0.1 + age * 0.12)
    elif trajectory == "declining":
        return max(0.2, 1.0 - age * 0.04)
    else:
        return 0.6 + min(0.4, age * 0.04)


def _build_yearly_tag_weights(years):
    """Pre-compute tag weight arrays for each year."""
    tag_names   = [t[0] for t in TAGS_RAW]
    base_counts = np.array([t[1] for t in TAGS_RAW], dtype=float)
    yearly = {}
    for year in years:
        mults = np.array([
            _tag_weight_at_year(t[0], t[2], t[3], year)
            for t in TAGS_RAW
        ])
        w = base_counts * mults
        yearly[year] = w / w.sum()
    return tag_names, yearly


def _random_tags(tag_names, weight_array):
    n = np.random.choice([1, 2, 3, 4, 5], p=[0.20, 0.35, 0.28, 0.12, 0.05])
    chosen = np.random.choice(tag_names, size=n, replace=False, p=weight_array)
    return "|".join(chosen)


def _realistic_date(start: datetime, end: datetime) -> datetime:
    """Pick a random datetime biased toward weekdays and daytime hours."""
    for _ in range(10):                      # try up to 10 times
        d = start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))
        if d.weekday() < 5:                  # Mon–Fri more likely
            break
    hour = int(np.random.choice(range(24), p=_hour_weights()))
    return d.replace(hour=hour, minute=random.randint(0, 59), second=0)


_hw = None
def _hour_weights():
    global _hw
    if _hw is None:
        raw = np.array([1,1,1,1,1,2,4,7,9,10,10,10,9,9,8,8,7,6,6,5,5,4,3,2], dtype=float)
        _hw = raw / raw.sum()
    return _hw


def _make_username(uid: int) -> str:
    prefixes  = ["dev", "code", "tech", "hack", "data", "cyber", "byte",
                 "algo", "stack", "null", "async", "git", "binary", "pixel"]
    suffixes  = ["wizard", "ninja", "guru", "master", "pro", "geek",
                 "coder", "dev", "hacker", "stack", "er"]
    if random.random() < 0.35:
        return f"{random.choice(prefixes)}{random.choice(suffixes)}{random.randint(10,9999)}"
    elif random.random() < 0.5:
        first = random.choice(["james","emma","liam","olivia","noah","ava",
                                "ethan","sophia","mason","isabella","arjun",
                                "priya","wei","hans","fatima","yuki"])
        last  = random.choice(["smith","jones","kumar","zhang","mueller",
                                "patel","kim","garcia","nguyen","ali"])
        return f"{first}_{last}{random.randint(0,999)}"
    else:
        return f"user_{uid}"


def _question_title(tags_str: str) -> str:
    tags = tags_str.split("|")
    tmpl_key = "_default"
    for t in tags:
        if t in TITLE_TEMPLATES:
            tmpl_key = t
            break
   
    if "python" in tags and "pandas" in tags:
        tmpl_key = "python+pandas"

    tmpl = random.choice(TITLE_TEMPLATES[tmpl_key])
    noun  = random.choice(_NOUNS)
    noun2 = random.choice(_NOUNS2)
    verb  = random.choice(_VERBS)
    tag_label = tags[0].capitalize()
    return (tmpl
            .replace("{verb}",      verb)
            .replace("{noun}",      noun)
            .replace("{noun2}",     noun2)
            .replace("{tag}",       tag_label))

# ─────────────────────────────────────────────────────────────────────────────
# Generators
# ─────────────────────────────────────────────────────────────────────────────

def generate_users():
    print("  Generating users …")
    rng = np.random.default_rng(42)

    reputations = np.clip(
        rng.pareto(1.5, N_USERS) * 100 + 1, 1, 500_000
    ).astype(int)

    # Tier labels drive downstream behaviour
    tiers = np.where(reputations >= 10_000, "expert",
            np.where(reputations >= 1_000,  "intermediate", "beginner"))

    loc_names, loc_probs = zip(*LOCATIONS)

    creation_dates = [
        datetime(2008, 1, 1) + timedelta(days=int(rng.integers(0, 5400)))
        for _ in range(N_USERS)
    ]

    # Badge counts correlated with reputation
    gold_count   = np.clip((reputations // 5_000), 0, 50)
    silver_count = np.clip((reputations // 500),  0, 200)
    bronze_count = np.clip((reputations // 50),   0, 500)

    df = pd.DataFrame({
        "Id":           np.arange(1, N_USERS + 1),
        "DisplayName":  [_make_username(i) for i in range(1, N_USERS + 1)],
        "Reputation":   reputations,
        "Tier":         tiers,
        "CreationDate": creation_dates,
        "Location":     rng.choice(loc_names, N_USERS,
                                   p=np.array(loc_probs) / np.sum(loc_probs)),
        "UpVotes":      rng.poisson(50, N_USERS),
        "DownVotes":    rng.poisson(5,  N_USERS),
        "GoldBadges":   gold_count,
        "SilverBadges": silver_count,
        "BronzeBadges": bronze_count,
        "LastAccessDate": [
            d + timedelta(days=int(rng.integers(0, (datetime(2024,1,1) - d).days + 1)))
            for d in creation_dates
        ],
    })
    df.to_csv("data/Users.csv", index=False)
    print(f"  {len(df):,} users saved.")
    return df


def generate_tags():
    print("  Generating tags …")
    df = pd.DataFrame({
        "TagName":    [t[0] for t in TAGS_RAW],
        "Count":      [t[1] for t in TAGS_RAW],
        "LaunchYear": [t[2] for t in TAGS_RAW],
        "Trajectory": [t[3] for t in TAGS_RAW],
    })
    df.to_csv("data/Tags.csv", index=False)
    print(f"  {len(df):,} tags saved.")


def generate_questions(users_df):
    print("  Generating questions …")

    START = datetime(2010, 1, 1)
    END   = datetime(2023, 12, 31)

    years = list(range(START.year, END.year + 1))
    tag_names, yearly_weights = _build_yearly_tag_weights(years)

    rng = np.random.default_rng(42)

    # Expert users post more high-score questions
    expert_ids       = users_df[users_df["Tier"] == "expert"]["Id"].values
    intermediate_ids = users_df[users_df["Tier"] == "intermediate"]["Id"].values
    beginner_ids     = users_df[users_df["Tier"] == "beginner"]["Id"].values

    rows = []
    for qid in range(1, N_QUESTIONS + 1):
        q_date = _realistic_date(START, END)
        year   = q_date.year
        tags   = _random_tags(tag_names, yearly_weights[year])

        tier_roll = rng.random()
        if tier_roll < 0.15:
            owner = int(rng.choice(expert_ids))
            score = int(rng.pareto(0.6) * 20)
        elif tier_roll < 0.40:
            owner = int(rng.choice(intermediate_ids))
            score = int(rng.pareto(0.9) * 7)
        else:
            owner = int(rng.choice(beginner_ids))
            score = int(rng.pareto(1.2) * 3)

        views = int(rng.pareto(0.6) * 500 + 10)
        answer_count = int(rng.choice(
            [0, 1, 2, 3, 4, 5], p=[0.18, 0.38, 0.26, 0.11, 0.05, 0.02]
        ))

        rows.append({
            "Id":               qid,
            "Title":            _question_title(tags),
            "Tags":             tags,
            "Score":            score,
            "ViewCount":        views,
            "AnswerCount":      answer_count,
            "CommentCount":     int(rng.poisson(1.5)),
            "OwnerUserId":      owner,
            "CreationDate":     q_date,
            "AcceptedAnswerId": (
                int(rng.integers(1, N_ANSWERS + 1))
                if rng.random() < 0.42 and answer_count > 0
                else np.nan
            ),
            "IsAnswered":       answer_count > 0,
            "DayOfWeek":        q_date.strftime("%A"),
            "YearMonth":        q_date.strftime("%Y-%m"),
        })

        if qid % 50_000 == 0:
            print(f"    … {qid:,} questions generated")

    df = pd.DataFrame(rows)
    df.to_csv("data/Questions.csv", index=False)
    print(f"  {len(df):,} questions saved.")
    return df


def generate_answers(questions_df, users_df):
    print("  Generating answers …")
    rng = np.random.default_rng(42)

    expert_ids       = users_df[users_df["Tier"] == "expert"]["Id"].values
    intermediate_ids = users_df[users_df["Tier"] == "intermediate"]["Id"].values
    beginner_ids     = users_df[users_df["Tier"] == "beginner"]["Id"].values

    answered = questions_df[questions_df["AnswerCount"] > 0].copy()
    rows, aid = [], 1

    for _, q in answered.iterrows():
        n_ans = int(q["AnswerCount"])
        for j in range(n_ans):
            # First answerer is more likely to be experienced
            tier_roll = rng.random()
            if j == 0 and tier_roll < 0.30:
                owner = int(rng.choice(expert_ids))
                score = int(rng.pareto(0.7) * 10)
            elif tier_roll < 0.45:
                owner = int(rng.choice(intermediate_ids))
                score = int(rng.pareto(0.9) * 5)
            else:
                owner = int(rng.choice(beginner_ids))
                score = max(0, int(rng.pareto(1.3) * 2))

            delay_hours = int(rng.integers(1, 24 * (j + 1) * 3 + 1))
            ans_date    = q["CreationDate"] + timedelta(hours=delay_hours)

            rows.append({
                "Id":           aid,
                "ParentId":     int(q["Id"]),
                "Score":        score,
                "OwnerUserId":  owner,
                "CreationDate": ans_date,
                "IsAccepted":   (j == 0 and not pd.isna(q["AcceptedAnswerId"])),
                "ResponseTimeHours": delay_hours,
            })
            aid += 1
            if aid > N_ANSWERS:
                break
        if aid > N_ANSWERS:
            break

    df = pd.DataFrame(rows)
    df.to_csv("data/Answers.csv", index=False)
    print(f"  {len(df):,} answers saved.")


def generate_badges(users_df):
    """Exploded badge event table — one row per badge earned."""
    print("  Generating badges …")
    rng    = np.random.default_rng(42)
    rows   = []

    for _, u in users_df.iterrows():
        uid      = int(u["Id"])
        join_dt  = u["CreationDate"]
        days_on  = max(1, (datetime(2024,1,1) - join_dt).days)

        for _ in range(int(u["GoldBadges"])):
            rows.append({"UserId": uid, "Class": "Gold",
                         "Name": random.choice(GOLD_BADGES),
                         "Date": join_dt + timedelta(days=int(rng.integers(0, days_on)))})
        for _ in range(min(int(u["SilverBadges"]), 20)):   # cap for file size
            rows.append({"UserId": uid, "Class": "Silver",
                         "Name": random.choice(SILVER_BADGES),
                         "Date": join_dt + timedelta(days=int(rng.integers(0, days_on)))})
        for _ in range(min(int(u["BronzeBadges"]), 20)):
            rows.append({"UserId": uid, "Class": "Bronze",
                         "Name": random.choice(BRONZE_BADGES),
                         "Date": join_dt + timedelta(days=int(rng.integers(0, days_on)))})

    df = pd.DataFrame(rows)
    df.to_csv("data/Badges.csv", index=False)
    print(f"  {len(df):,} badge events saved.")




def run():
    np.random.seed(42)
    random.seed(42)
    os.makedirs("data", exist_ok=True)

    print("\n[generate_data] Starting …")
    users_df     = generate_users()
    generate_tags()
    questions_df = generate_questions(users_df)
    generate_answers(questions_df, users_df)
    generate_badges(users_df)
    print("[generate_data] All done.\n")


if __name__ == "__main__":
    run()