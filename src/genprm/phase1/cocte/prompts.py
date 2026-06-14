"""Prompt templates for LLM-backed CoCTE transformation (RewardSQL Appendix A.2)."""

COCTE_TRANSFORM_PROMPT = """You are tasked with rewriting the following SQL query using linear Common Table Expressions (CTEs) and annotating the rationale behind each step.

Rules:
1. Decompose the query into a **linear** chain of independent CTEs — no nested subqueries inside CTE bodies when avoidable.
2. Each CTE must be named descriptively (PascalCase or snake_case).
3. Later CTEs may reference earlier CTEs only (never forward references).
4. The final answer must be a standalone SELECT (not wrapped in WITH).
5. Provide a brief rationale for each CTE explaining the reasoning step.

Output JSON with this schema:
{{
  "steps": [
    {{"cte_name": "...", "query": "SELECT ...", "rationale": "..."}},
    ...
  ],
  "final_query": "SELECT ..."
}}

{few_shot_examples}

Problem:
- Question: {question}
- Schema: {schema}
- SQL: {gold_sql}
"""

FEW_SHOT_EXAMPLES = """
Example:
- Question: Please list the team names which have at least 3 all-star players.
- SQL: SELECT players_teams.tmid FROM players_teams INNER JOIN player_allstar ON players_teams.playerid = player_allstar.playerid GROUP BY players_teams.tmid HAVING count(DISTINCT players_teams.playerid) >= 3
- Output:
  steps:
    - cte_name: All_Star_Players
      query: SELECT playerid FROM player_allstar
      rationale: Identify all all-star players first.
    - cte_name: All_Star_Team_Associations
      query: SELECT pt.tmid, pt.playerid FROM players_teams AS pt INNER JOIN All_Star_Players AS asp ON pt.playerid = asp.playerid
      rationale: Map all-star players to their teams.
    - cte_name: Teams_With_Three_All_Stars
      query: SELECT at.tmid, COUNT(DISTINCT at.playerid) AS all_star_count FROM All_Star_Team_Associations AS at GROUP BY at.tmid HAVING COUNT(DISTINCT at.playerid) >= 3
      rationale: Filter teams with at least three distinct all-star players.
  final_query: SELECT t.name FROM Teams_With_Three_All_Stars AS twa INNER JOIN teams AS t ON twa.tmid = t.tmid
"""

SCHEMA_FILTER_PROMPT = """Evaluate the relevance of columns in the database schema to the user's question, selecting the minimal necessary column set.

Schema analysis steps:
- Identify entities and calculation metrics in the question
- Match relevant tables through primary/foreign keys
- Filter columns directly related to query conditions
- Retain numeric fields required for calculations

Current task schema:
{schema}

Question: {question}
Evidence: {evidence}

Return JSON: {{"tables": [...], "columns": {{"table_name": ["col1", ...]}}}}
"""
