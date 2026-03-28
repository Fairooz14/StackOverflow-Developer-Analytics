import os
import duckdb
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# All queries use DuckDB SQL.  Views available:
#   questions  – Id, Title, Tags, Score, ViewCount, AnswerCount, CommentCount,
#                OwnerUserId, CreationDate, AcceptedAnswerId, IsAnswered,
#                DayOfWeek, YearMonth
#   answers    – Id, ParentId, Score, OwnerUserId, CreationDate, IsAccepted,
#                ResponseTimeHours
#   users      – Id, DisplayName, Reputation, Tier, CreationDate, Location,
#                UpVotes, DownVotes, GoldBadges, SilverBadges, BronzeBadges,
#                LastAccessDate
#   tags       – TagName, Count, LaunchYear, Trajectory
#   badges     – UserId, Class, Name, Date
# ─────────────────────────────────────────────────────────────────────────────

QUERIES = {

    # ── 01 · Tag popularity with momentum score ───────────────────────────────
    # Combines raw volume with recency-weighted growth to produce a single
    # "momentum" rank — more nuanced than a plain count.
    "Q01_tag_momentum": """
        WITH yearly AS (
            SELECT TRIM(tag)           AS tag,
                   YEAR(q.CreationDate) AS yr,
                   COUNT(*)            AS cnt
            FROM questions q,
                 UNNEST(STRING_SPLIT(q.Tags, '|')) AS t(tag)
            WHERE tag != ''
              AND YEAR(q.CreationDate) BETWEEN 2018 AND 2023
            GROUP BY tag, yr
        ),
        pivoted AS (
            SELECT tag,
                   SUM(cnt)                                    AS total_6yr,
                   SUM(cnt) FILTER (WHERE yr >= 2021)          AS recent_3yr,
                   SUM(cnt) FILTER (WHERE yr  < 2021)          AS older_3yr,
                   MAX(cnt) FILTER (WHERE yr = 2023)           AS cnt_2023,
                   MAX(cnt) FILTER (WHERE yr = 2022)           AS cnt_2022
            FROM yearly
            GROUP BY tag
        )
        SELECT tag,
               total_6yr,
               recent_3yr,
               older_3yr,
               ROUND(100.0 * recent_3yr / NULLIF(older_3yr, 0), 1) AS growth_pct,
               -- weighted momentum: 60% recent share + 40% yoy acceleration
               ROUND(
                   0.6 * (recent_3yr * 1.0 / NULLIF(total_6yr, 0)) +
                   0.4 * (cnt_2023   * 1.0 / NULLIF(cnt_2022,  0)),
               4) AS momentum_score
        FROM pivoted
        WHERE total_6yr > 200
        ORDER BY momentum_score DESC
        LIMIT 25
    """,

    # ── 02 · Response-time percentiles per tag ────────────────────────────────
    # P50 / P90 / P99 response time reveals not just the average but the tail
    # behaviour — a classic SRE-style metric applied to community health.
    "Q02_response_time_percentiles": """
        WITH first_ans AS (
            SELECT q.Id,
                   TRIM(tag)              AS tag,
                   a.ResponseTimeHours    AS hrs
            FROM questions q
            JOIN answers a ON a.ParentId = q.Id AND a.IsAccepted = true
            ,    UNNEST(STRING_SPLIT(q.Tags, '|')) AS t(tag)
            WHERE tag != ''
        )
        SELECT tag,
               COUNT(*)                              AS accepted_answers,
               ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY hrs), 1) AS p25_hrs,
               ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY hrs), 1) AS p50_hrs,
               ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY hrs), 1) AS p75_hrs,
               ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY hrs), 1) AS p90_hrs,
               ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY hrs), 1) AS p99_hrs,
               ROUND(AVG(hrs), 1)                   AS mean_hrs
        FROM first_ans
        GROUP BY tag
        HAVING accepted_answers > 200
        ORDER BY p50_hrs ASC
        LIMIT 25
    """,

    # ── 03 · Monthly question volume + 3-month rolling average ───────────────
    # Rolling average smooths noise and shows true trend for a time-series chart.
    "Q03_monthly_rolling_avg": """
        WITH monthly AS (
            SELECT YearMonth,
                   YEAR(CreationDate)   AS yr,
                   MONTH(CreationDate)  AS mo,
                   COUNT(*)             AS questions,
                   ROUND(AVG(Score), 2) AS avg_score,
                   SUM(ViewCount)       AS total_views
            FROM questions
            GROUP BY YearMonth, yr, mo
        )
        SELECT YearMonth, yr, mo, questions, avg_score, total_views,
               ROUND(AVG(questions) OVER (
                   ORDER BY yr, mo
                   ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
               ), 1) AS rolling_3m_questions,
               ROUND(AVG(avg_score) OVER (
                   ORDER BY yr, mo
                   ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
               ), 2) AS rolling_3m_score
        FROM monthly
        ORDER BY yr, mo
    """,

    # ── 04 · Full engagement funnel per tag ───────────────────────────────────
    # Views → question posted → answered → accepted: measures community
    # engagement depth and where questions "drop off".
    "Q04_engagement_funnel": """
        WITH base AS (
            SELECT TRIM(tag) AS tag,
                   Id, AnswerCount, AcceptedAnswerId, ViewCount, Score
            FROM questions,
                 UNNEST(STRING_SPLIT(Tags, '|')) AS t(tag)
            WHERE tag IN (
                'python','javascript','sql','java','react',
                'typescript','go','rust','django','flask',
                'machine-learning','docker','kubernetes','postgresql'
            )
        )
        SELECT tag,
               COUNT(*)                                                         AS total_q,
               SUM(ViewCount)                                                   AS total_views,
               ROUND(SUM(ViewCount) * 1.0 / COUNT(*), 0)                       AS views_per_q,
               COUNT(*) FILTER (WHERE AnswerCount > 0)                         AS answered,
               COUNT(*) FILTER (WHERE AcceptedAnswerId IS NOT NULL)            AS accepted,
               ROUND(100.0 * COUNT(*) FILTER (WHERE AnswerCount > 0)
                     / COUNT(*), 1)                                             AS answer_rate_pct,
               ROUND(100.0 * COUNT(*) FILTER (WHERE AcceptedAnswerId IS NOT NULL)
                     / NULLIF(COUNT(*) FILTER (WHERE AnswerCount > 0), 0), 1)  AS accept_of_answered_pct,
               ROUND(AVG(Score), 2)                                            AS avg_score
        FROM base
        GROUP BY tag
        ORDER BY total_q DESC
    """,

    # ── 05 · Expert user contribution analysis ────────────────────────────────
    # What fraction of all high-quality content (score ≥ 10) comes from the
    # top 1% of users by reputation? Classic "power law" portfolio analysis.
    "Q05_expert_contribution_share": """
        WITH rep_ranked AS (
            SELECT Id,
                   DisplayName,
                   Reputation,
                   Tier,
                   NTILE(100) OVER (ORDER BY Reputation) AS rep_percentile
            FROM users
        ),
        q_scored AS (
            SELECT q.OwnerUserId,
                   COUNT(*)                             AS total_q,
                   COUNT(*) FILTER (WHERE q.Score >= 5) AS high_score_q,
                   SUM(q.Score)                         AS score_sum
            FROM questions q
            GROUP BY q.OwnerUserId
        )
        SELECT r.rep_percentile,
               COUNT(DISTINCT r.Id)           AS user_count,
               SUM(qs.total_q)                AS total_questions,
               SUM(qs.high_score_q)           AS high_score_questions,
               SUM(qs.score_sum)              AS total_score,
               ROUND(AVG(r.Reputation), 0)    AS avg_reputation,
               -- share of all platform high-score questions
               ROUND(100.0 * SUM(qs.high_score_q)
                     / SUM(SUM(qs.high_score_q)) OVER (), 2) AS pct_of_all_hq
        FROM rep_ranked r
        LEFT JOIN q_scored qs ON qs.OwnerUserId = r.Id
        GROUP BY r.rep_percentile
        ORDER BY r.rep_percentile DESC
        LIMIT 20
    """,

    # ── 06 · Year-over-year tag growth with rank change ───────────────────────
    # Shows not just growth % but whether a tag climbed or fell in the ranking —
    # useful for a "mover & shaker" leaderboard visual.
    "Q06_yoy_rank_change": """
        WITH yearly AS (
            SELECT YEAR(q.CreationDate) AS yr,
                   TRIM(tag)            AS tag,
                   COUNT(*)             AS cnt
            FROM questions q,
                 UNNEST(STRING_SPLIT(q.Tags, '|')) AS t(tag)
            WHERE tag != ''
              AND YEAR(q.CreationDate) BETWEEN 2015 AND 2023
            GROUP BY yr, tag
        ),
        ranked AS (
            SELECT yr, tag, cnt,
                   RANK() OVER (PARTITION BY yr  ORDER BY cnt DESC) AS yr_rank,
                   LAG(cnt) OVER (PARTITION BY tag ORDER BY yr)     AS prev_cnt
            FROM yearly
        ),
        with_prev_rank AS (
            SELECT yr, tag, cnt, yr_rank, prev_cnt,
                   LAG(yr_rank) OVER (PARTITION BY tag ORDER BY yr) AS prev_rank
            FROM ranked
        )
        SELECT yr, tag, cnt AS question_count, yr_rank,
               COALESCE(prev_rank - yr_rank, 0)                          AS rank_change,
               ROUND(100.0 * (cnt - prev_cnt) / NULLIF(prev_cnt, 0), 1) AS yoy_pct
        FROM with_prev_rank
        WHERE yr IN (2019, 2020, 2021, 2022, 2023)
          AND yr_rank <= 20
        ORDER BY yr DESC, yr_rank
    """,

    # ── 07 · Unanswered high-visibility questions ─────────────────────────────
    # Questions with many views but no accepted answer represent unmet demand —
    # an interesting "opportunity gap" metric.
    "Q07_opportunity_gap": """
        SELECT q.Id,
               q.Title,
               TRIM(STRING_SPLIT(q.Tags, '|')[1]) AS primary_tag,
               q.ViewCount,
               q.Score,
               q.AnswerCount,
               q.CommentCount,
               YEAR(q.CreationDate) AS year,
               -- opportunity score: high views, low answers, high score
               ROUND(
                   LOG(q.ViewCount + 1) * (q.Score + 1)
                   / (q.AnswerCount + 1),
               2) AS opportunity_score
        FROM questions q
        WHERE q.AcceptedAnswerId IS NULL
          AND q.ViewCount > 3000
        ORDER BY opportunity_score DESC
        LIMIT 40
    """,

    # ── 08 · Heatmap of posting activity (day × hour) ─────────────────────────
    # Feeds directly into a Plotly/D3 heatmap tile.
    "Q08_activity_heatmap": """
        SELECT DayOfWeek,
               HOUR(CreationDate)    AS hour_of_day,
               COUNT(*)              AS question_count,
               ROUND(AVG(Score), 2)  AS avg_score,
               ROUND(AVG(ViewCount), 0) AS avg_views
        FROM questions
        GROUP BY DayOfWeek, hour_of_day
        ORDER BY
            CASE DayOfWeek
                WHEN 'Monday'    THEN 1 WHEN 'Tuesday'   THEN 2
                WHEN 'Wednesday' THEN 3 WHEN 'Thursday'  THEN 4
                WHEN 'Friday'    THEN 5 WHEN 'Saturday'  THEN 6
                ELSE 7
            END,
            hour_of_day
    """,

    # ── 09 · User cohort retention (join-year cohorts) ────────────────────────
    # For each cohort of users (by join year), how many were still posting
    # questions 1, 2, 3 years later?  Classic product retention analysis.
    "Q09_cohort_retention": """
        WITH cohorts AS (
            SELECT u.Id                        AS uid,
                   YEAR(u.CreationDate)         AS cohort_year,
                   YEAR(q.CreationDate)         AS active_year
            FROM users u
            JOIN questions q ON q.OwnerUserId = u.Id
        ),
        base AS (
            SELECT cohort_year,
                   COUNT(DISTINCT uid) AS cohort_size
            FROM cohorts
            GROUP BY cohort_year
        ),
        yearly_active AS (
            SELECT cohort_year,
                   active_year,
                   active_year - cohort_year AS years_since_join,
                   COUNT(DISTINCT uid)       AS active_users
            FROM cohorts
            GROUP BY cohort_year, active_year
        )
        SELECT ya.cohort_year,
               b.cohort_size,
               ya.years_since_join,
               ya.active_users,
               ROUND(100.0 * ya.active_users / b.cohort_size, 1) AS retention_pct
        FROM yearly_active ya
        JOIN base b USING (cohort_year)
        WHERE ya.cohort_year BETWEEN 2010 AND 2020
          AND ya.years_since_join BETWEEN 0 AND 5
        ORDER BY ya.cohort_year, ya.years_since_join
    """,

    # ── 10 · Answer quality by user tier and tag category ────────────────────
    # Cross-tab: do expert users consistently produce better answers on
    # certain tag families?  Useful for a grouped bar chart.
    "Q10_answer_quality_by_tier_tag": """
        WITH tag_family AS (
            SELECT a.Id        AS answer_id,
                   a.Score     AS a_score,
                   a.IsAccepted,
                   a.ResponseTimeHours,
                   u.Tier,
                   u.Reputation,
                   TRIM(tag)   AS tag
            FROM answers a
            JOIN users   u ON u.Id = a.OwnerUserId
            JOIN questions q ON q.Id = a.ParentId
            ,    UNNEST(STRING_SPLIT(q.Tags, '|')) AS t(tag)
            WHERE tag IN (
                'python','javascript','java','sql','react',
                'machine-learning','docker','go','rust','typescript'
            )
        )
        SELECT tag,
               Tier,
               COUNT(*)                                    AS answer_count,
               ROUND(AVG(a_score), 2)                     AS avg_score,
               ROUND(AVG(ResponseTimeHours), 1)           AS avg_response_hrs,
               ROUND(100.0 * SUM(IsAccepted::INT)
                     / COUNT(*), 1)                        AS accept_rate_pct,
               ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
                     (ORDER BY a_score), 1)                AS median_score
        FROM tag_family
        GROUP BY tag, Tier
        HAVING answer_count > 50
        ORDER BY tag, Tier
    """,

    # ── 11 · Badge velocity: how fast do users earn badges? ──────────────────
    # Days-to-first-badge grouped by reputation tier — shows onboarding health.
    "Q11_badge_velocity": """
        WITH first_badge AS (
            SELECT b.UserId,
                   MIN(b.Date)          AS first_badge_date,
                   COUNT(*)             AS total_badges,
                   COUNT(*) FILTER (WHERE b.Class = 'Gold')   AS gold,
                   COUNT(*) FILTER (WHERE b.Class = 'Silver') AS silver,
                   COUNT(*) FILTER (WHERE b.Class = 'Bronze') AS bronze
            FROM badges b
            GROUP BY b.UserId
        ),
        joined AS (
            SELECT u.Tier,
                   u.Reputation,
                   DATE_DIFF('day', u.CreationDate, fb.first_badge_date) AS days_to_first,
                   fb.total_badges,
                   fb.gold, fb.silver, fb.bronze
            FROM users u
            JOIN first_badge fb ON fb.UserId = u.Id
            WHERE DATE_DIFF('day', u.CreationDate, fb.first_badge_date) >= 0
        )
        SELECT Tier,
               COUNT(*)                                AS users_with_badge,
               ROUND(AVG(days_to_first), 1)            AS avg_days_to_first_badge,
               ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
                     (ORDER BY days_to_first), 0)      AS median_days,
               ROUND(AVG(total_badges), 1)             AS avg_total_badges,
               ROUND(AVG(gold),   2)                   AS avg_gold,
               ROUND(AVG(silver), 2)                   AS avg_silver,
               ROUND(AVG(bronze), 2)                   AS avg_bronze
        FROM joined
        GROUP BY Tier
        ORDER BY avg_days_to_first_badge
    """,

    # ── 12 · Geographic productivity index ───────────────────────────────────
    # Score-per-user by country — normalises for user count so small but highly
    # productive communities surface (not just "India has many users").
    "Q12_geo_productivity": """
        WITH user_scores AS (
            SELECT u.Location,
                   u.Id,
                   u.Reputation,
                   u.Tier,
                   COALESCE(SUM(q.Score), 0) AS q_score
            FROM users u
            LEFT JOIN questions q ON q.OwnerUserId = u.Id
            WHERE u.Location != ''
            GROUP BY u.Location, u.Id, u.Reputation, u.Tier
        )
        SELECT Location,
               COUNT(DISTINCT Id)                       AS user_count,
               COUNT(DISTINCT Id) FILTER
                   (WHERE Tier = 'expert')               AS expert_count,
               ROUND(AVG(Reputation), 0)                AS avg_reputation,
               ROUND(AVG(q_score), 2)                   AS avg_q_score_per_user,
               ROUND(SUM(q_score) * 1.0 / COUNT(DISTINCT Id), 2)
                                                         AS productivity_index,
               ROUND(100.0 * COUNT(DISTINCT Id) FILTER (WHERE Tier = 'expert')
                     / COUNT(DISTINCT Id), 1)            AS expert_pct
        FROM user_scores
        GROUP BY Location
        HAVING user_count >= 50
        ORDER BY productivity_index DESC
        LIMIT 20
    """,

    # ── 13 · Trajectory cohort: rising vs declining tags ─────────────────────
    # Joins the Tags metadata (Trajectory column from generate_data) with
    # actual question volume to validate whether synthetic trajectories match.
    "Q13_trajectory_validation": """
        WITH yearly AS (
            SELECT TRIM(tag)            AS tag,
                   YEAR(q.CreationDate)  AS yr,
                   COUNT(*)              AS cnt
            FROM questions q,
                 UNNEST(STRING_SPLIT(q.Tags, '|')) AS t(tag)
            WHERE tag != ''
              AND YEAR(q.CreationDate) BETWEEN 2012 AND 2023
            GROUP BY tag, yr
        ),
        with_meta AS (
            SELECT y.tag, y.yr, y.cnt,
                   tg.Trajectory,
                   tg.LaunchYear,
                   FIRST_VALUE(y.cnt) OVER
                       (PARTITION BY y.tag ORDER BY y.yr) AS base_cnt
            FROM yearly y
            JOIN tags tg ON tg.TagName = y.tag
        )
        SELECT tag, Trajectory, LaunchYear, yr,
               cnt,
               ROUND(100.0 * (cnt - base_cnt) / NULLIF(base_cnt, 0), 1)
                   AS pct_change_from_base
        FROM with_meta
        ORDER BY Trajectory, tag, yr
    """,

    # ── 14 · Power users: multi-dimensional leaderboard ──────────────────────
    # Combines questions, answers, badges, and reputation into a composite
    # "platform value" score — great for a leaderboard table visual.
    "Q14_power_user_leaderboard": """
        WITH q_stats AS (
            SELECT OwnerUserId,
                   COUNT(*)                              AS q_count,
                   SUM(Score)                            AS q_score_total,
                   COUNT(*) FILTER (WHERE Score >= 10)  AS viral_questions
            FROM questions
            GROUP BY OwnerUserId
        ),
        a_stats AS (
            SELECT OwnerUserId,
                   COUNT(*)                              AS a_count,
                   SUM(Score)                            AS a_score_total,
                   SUM(IsAccepted::INT)                  AS accepted_answers
            FROM answers
            GROUP BY OwnerUserId
        ),
        b_stats AS (
            SELECT UserId,
                   SUM(CASE Class WHEN 'Gold' THEN 10
                                  WHEN 'Silver' THEN 3
                                  ELSE 1 END)            AS badge_points
            FROM badges
            GROUP BY UserId
        )
        SELECT u.Id,
               u.DisplayName,
               u.Reputation,
               u.Tier,
               u.Location,
               COALESCE(q.q_count, 0)          AS questions,
               COALESCE(a.a_count, 0)          AS answers,
               COALESCE(q.viral_questions, 0)  AS viral_questions,
               COALESCE(a.accepted_answers, 0) AS accepted_answers,
               COALESCE(b.badge_points, 0)     AS badge_points,
               -- composite platform value score
               ROUND(
                   COALESCE(q.q_score_total, 0) * 1.0
                   + COALESCE(a.a_score_total, 0) * 1.5
                   + COALESCE(b.badge_points, 0) * 2.0
                   + u.Reputation * 0.01,
               2) AS platform_value_score
        FROM users u
        LEFT JOIN q_stats q ON q.OwnerUserId = u.Id
        LEFT JOIN a_stats a ON a.OwnerUserId = u.Id
        LEFT JOIN b_stats b ON b.UserId = u.Id
        WHERE COALESCE(q.q_count, 0) + COALESCE(a.a_count, 0) > 0
        ORDER BY platform_value_score DESC
        LIMIT 30
    """,

    # ── 15 · Self-answer rate and "quick close" questions ────────────────────
    # What % of questions are answered by the same user who asked?
    # High self-answer rate on a tag might mean "documentation gap" questions.
    "Q15_self_answer_rate": """
        WITH self_ans AS (
            SELECT q.Id,
                   TRIM(tag)                       AS tag,
                   q.OwnerUserId                   AS asker,
                   q.Score                         AS q_score,
                   q.ViewCount,
                   MAX(CASE WHEN a.OwnerUserId = q.OwnerUserId
                            THEN 1 ELSE 0 END)     AS self_answered,
                   MAX(CASE WHEN a.IsAccepted
                            THEN 1 ELSE 0 END)     AS has_accepted
            FROM questions q
            JOIN answers a ON a.ParentId = q.Id
            ,    UNNEST(STRING_SPLIT(q.Tags, '|')) AS t(tag)
            WHERE tag IN (
                'python','javascript','java','sql','react',
                'typescript','go','rust','docker','kubernetes',
                'machine-learning','postgresql','django','flask'
            )
            GROUP BY q.Id, tag, q.OwnerUserId, q.Score, q.ViewCount
        )
        SELECT tag,
               COUNT(*)                                     AS total_answered,
               SUM(self_answered)                           AS self_answered_count,
               ROUND(100.0 * SUM(self_answered)
                     / COUNT(*), 1)                         AS self_answer_rate_pct,
               ROUND(AVG(q_score), 2)                       AS avg_q_score,
               ROUND(AVG(ViewCount), 0)                     AS avg_views
        FROM self_ans
        GROUP BY tag
        HAVING total_answered > 100
        ORDER BY self_answer_rate_pct DESC
    """,
}


def run():
    os.makedirs("results", exist_ok=True)

    print("[queries] Connecting to DuckDB …")
    con = duckdb.connect()
    con.execute("CREATE VIEW questions AS SELECT * FROM read_csv_auto('data/Questions.csv')")
    con.execute("CREATE VIEW answers   AS SELECT * FROM read_csv_auto('data/Answers.csv')")
    con.execute("CREATE VIEW users     AS SELECT * FROM read_csv_auto('data/Users.csv')")
    con.execute("CREATE VIEW tags      AS SELECT * FROM read_csv_auto('data/Tags.csv')")
    con.execute("CREATE VIEW badges    AS SELECT * FROM read_csv_auto('data/Badges.csv')")

    print(f"[queries] Running {len(QUERIES)} queries …\n")
    failed = []
    for name, sql in QUERIES.items():
        try:
            df = con.execute(sql).fetchdf()
            df.to_csv(f"results/{name}.csv", index=False)
            print(f"  {name:<40} {len(df):>6,} rows")
        except Exception as e:
            print(f"  {name:<40} ERROR: {e}")
            failed.append(name)

    con.close()
    print(f"\n[queries] Done.  {len(QUERIES) - len(failed)}/{len(QUERIES)} succeeded.\n")
    if failed:
        print("  Failed queries:", failed)


if __name__ == "__main__":
    run()