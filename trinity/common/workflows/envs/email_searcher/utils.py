'''
This file defines Email Dataclass and three email_search_tools.
Modified from https://github.com/OpenPipe/ART/blob/art-e/examples/art-e/
'''
import os
import sqlite3
import subprocess
import openai
import json
from typing import Any, Dict, List, Optional
from dataclasses import asdict, dataclass, field, fields
from pydantic import BaseModel

from trinity.utils.log import get_logger

from .prepare_data import DEFAULT_DB_PATH

logger = get_logger(__name__)

RUBRIC_FILE = "rubric_log.jsonl" # TODO: pass from outside

conn = None
def get_conn():
    global conn
    if conn is None:
        conn = sqlite3.connect(
            f"file:{DEFAULT_DB_PATH}?mode=ro", uri=True, check_same_thread=False
        )
    return conn

############ Define dataclass ############

class Query(BaseModel):
    id: int
    question: str
    answer: str
    message_ids: List[str]  # message_ids (strings) of referenced emails
    how_realistic: float
    inbox_address: str
    query_date: str


class Email(BaseModel):
    message_id: str
    date: str  # ISO 8601 string 'YYYY-MM-DD HH:MM:SS'
    subject: Optional[str] = None
    from_address: Optional[str] = None
    to_addresses: List[str] = []  # Populated from recipients table
    cc_addresses: List[str] = []  # Populated from recipients table
    bcc_addresses: List[str] = []  # Populated from recipients table
    body: Optional[str] = None
    file_name: Optional[str] = None


@dataclass
class SearchResult:
    message_id: str
    snippet: str


class FinalRubric(BaseModel):
    answer_correct: bool = False
    sources_correct: bool = False
    num_turns: int = 0
    attempted_answer: bool = False
    ever_found_right_email: bool = False
    ever_read_right_email: bool = False
    cant_parse_tool_call: bool = False
    bad_tool_call_name: bool = False
    bad_tool_call_args: bool = False
    ran_out_of_turns: bool = False
    returned_i_dont_know: bool = False
    num_sources: int = 0
    ever_tried_to_read_invalid_email: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    sources: List[str] = []

    def to_metrics(self) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {k: int(v) for k, v in asdict(self).items()}
        metrics["failed_format_validation"] = int(
            self.bad_tool_call_name or self.bad_tool_call_args or self.cant_parse_tool_call
        )
        return metrics


############ Define tools for agentscope ############

def search_emails(query: Any, keywords: list[str], timeout: float = 300, **kwargs: Any):
    """
    Search the user's email inbox for emails that match the given keywords.

    Args:
        query (Any): The query object containing the user's inbox address, query date, and message IDs.
        keywords (list[str]): A list of keywords to search for in the user's email inbox.

    Returns:
        ServiceResponse: 
            The status field indicates whether the tool call was successful. 
            The content field contains a list of SearchResult objects with message_id and snippet.
    """
    from agentscope.service import ServiceResponse, ServiceExecStatus
    
    ever_found_right_email = False
    try:
        res = subprocess.run(
            search_emails_tool(
                inbox=query.inbox_address,
                sent_before=query.query_date,
                keywords=keywords
                ),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        
        for r in res:
            if r.message_id == query.message_ids[0]:
                ever_found_right_email = True

        # save to rubric file
        current_state = get_rubric_state()
        new_state = current_state.copy()
        new_state["ever_found_right_email"] = ever_found_right_email
        save_rubric_state(new_state)

        return ServiceResponse(
                status=ServiceExecStatus.SUCCESS,
                content=[asdict(r) for r in res],
            )
    except subprocess.TimeoutExpired:
        return ServiceResponse(
            status=ServiceExecStatus.ERROR,
            content=None,
        )


def read_email(query: Any, message_id: str, timeout: float = 300, **kwargs: Any):
    """
    Read the content of an email from the user's email inbox. Returns the email content.
    Args:
        message_id (str): The unique identifier of the email to read.

    Returns:
        ServiceResponse:
            The status field indicates whether the tool call was successful.
            The content field contains the email content or an error message if the email is not found.
    """
    from agentscope.service import ServiceResponse, ServiceExecStatus

    ever_found_right_email = False
    try:
        email_content = subprocess.run(
            read_email_tool(message_id),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )

        if message_id == query.message_ids[0]:
            ever_read_right_email = True
        
        # save to rubric file
        current_state = get_rubric_state()
        new_state = current_state.copy()
        new_state["ever_read_right_email"] = ever_read_right_email
        save_rubric_state(new_state)

        if email_content is None:
            return ServiceResponse(
                status=ServiceExecStatus.ERROR,
                content={"error": "Email not found"},
            )
        else:
            return ServiceResponse(
                status=ServiceExecStatus.SUCCESS,
                content={"email_content": email_content.model_dump()},
            )
    except subprocess.TimeoutExpired:
        return ServiceResponse(
            status=ServiceExecStatus.ERROR,
            content={"error": "Timeout", "ever_found_right_email": ever_found_right_email},
        )


def search_emails_tool(
    inbox: str,
    keywords: List[str],
    from_addr: Optional[str] = None,
    to_addr: Optional[str] = None,
    sent_after: Optional[str] = None,
    sent_before: Optional[str] = None,
    max_results: int = 10,
) -> List[SearchResult]:
    """
    Searches the email database based on keywords, inbox, sender, recipient, and date range.

    Args:
        inbox: The email address of the user performing the search.
               Results include emails sent from or to (inc. cc/bcc) this address.
        keywords: A list of keywords that must all appear in the subject or body.
        from_addr: Optional email address to filter emails sent *from*.
        to_addr: Optional email address to filter emails sent *to* (inc. cc/bcc).
        sent_after: Optional date string 'YYYY-MM-DD'. Filters for emails sent on or after this date.
        sent_before: Optional date string 'YYYY-MM-DD'. Filters for emails sent before this date.
        max_results: The maximum number of results to return. Cannot exceed 10.

    Returns:
        A list of SearchResult objects, each containing 'message_id' and 'snippet'.
        Returns an empty list if no results are found or an error occurs.
    """

    # Initialize sql and params
    sql: Optional[str] = None
    params: List[str | int] = []

    cursor = get_conn().cursor()

    # --- Build Query ---
    where_clauses: List[str] = []

    # 1. Keywords (FTS)
    if not keywords:
        raise ValueError("No keywords provided for search.")

    if max_results > 10:
        raise ValueError("max_results must be less than or equal to 10.")

    # FTS5 default is AND, so just join keywords. Escape quotes for safety.
    fts_query = " ".join(f""" "{k.replace('"', '""')}" """ for k in keywords)
    where_clauses.append("fts.emails_fts MATCH ?")
    params.append(fts_query)

    # 2. Inbox filter (must be from OR to/cc/bcc the inbox user)
    # Use the composite index idx_recipients_address_email here
    where_clauses.append(
        """
        (e.from_address = ? OR EXISTS (
            SELECT 1 FROM recipients r_inbox
            WHERE r_inbox.recipient_address = ? AND r_inbox.email_id = e.id
        ))
        """
    )
    params.extend([inbox, inbox])

    # 3. Optional From filter
    if from_addr:
        where_clauses.append("e.from_address = ?")
        params.append(from_addr)

    # 4. Optional To filter (includes to, cc, bcc)
    # Use the composite index idx_recipients_address_email here
    if to_addr:
        where_clauses.append(
            """
            EXISTS (
                SELECT 1 FROM recipients r_to
                WHERE r_to.recipient_address = ? AND r_to.email_id = e.id
            )
            """
        )
        params.append(to_addr)

    # 5. Optional Sent After filter
    if sent_after:
        # Assumes date format 'YYYY-MM-DD'
        # Compare against the start of the day
        where_clauses.append("e.date >= ?")
        params.append(f"{sent_after} 00:00:00")

    # 6. Optional Sent Before filter
    if sent_before:
        # Assumes date format 'YYYY-MM-DD'
        # Compare against the start of the day (exclusive)
        where_clauses.append("e.date < ?")
        params.append(f"{sent_before} 00:00:00")

    # --- Construct Final Query ---
    # snippet(<table>, <column_index>, <highlight_start>, <highlight_end>, <ellipsis>, <tokens>)
    # -1 means highlight across all columns (subject, body)
    sql = f"""
        SELECT
            e.message_id,
            snippet(emails_fts, -1, '<b>', '</b>', ' ... ', 15) as snippet
        FROM
            emails e JOIN emails_fts fts ON e.id = fts.rowid
        WHERE
            {" AND ".join(where_clauses)}
        ORDER BY
            e.date DESC -- Order by date for relevance
        LIMIT ?;
    """
    params.append(max_results)

    # --- Execute and Fetch ---
    logger.debug(f"Executing SQL: {sql}")
    logger.debug(f"With params: {params}")
    cursor.execute(sql, params)
    results = cursor.fetchall()

    # Format results
    formatted_results = [
        SearchResult(message_id=row[0], snippet=row[1]) for row in results
    ]
    logger.info(f"Search found {len(formatted_results)} results.")
    return formatted_results


def read_email_tool(message_id: str) -> Optional[Email]:
    """
    Retrieves a single email by its message_id from the database.

    Args:
        message_id: The unique identifier of the email to retrieve.

    Returns:
        An Email object containing the details of the found email,
        or None if the email is not found or an error occurs.
    """

    cursor = get_conn().cursor()

    # --- Query for Email Core Details ---
    email_sql = """
        SELECT message_id, date, subject, from_address, body, file_name
        FROM emails
        WHERE message_id = ?;
    """
    cursor.execute(email_sql, (message_id,))
    email_row = cursor.fetchone()

    if not email_row:
        logger.warning(f"Email with message_id '{message_id}' not found.")
        return None

    (
        msg_id,
        date,
        subject,
        from_addr,
        body,
        file_name,
    ) = email_row

    # --- Query for Recipients ---
    recipients_sql = """
        SELECT recipient_address, recipient_type
        FROM recipients
        WHERE email_id = ?;
    """
    cursor.execute(recipients_sql, (message_id,))
    recipient_rows = cursor.fetchall()

    to_addresses: List[str] = []
    cc_addresses: List[str] = []
    bcc_addresses: List[str] = []

    for addr, type in recipient_rows:
        type_lower = type.lower()
        if type_lower == "to":
            to_addresses.append(addr)
        elif type_lower == "cc":
            cc_addresses.append(addr)
        elif type_lower == "bcc":
            bcc_addresses.append(addr)

    # --- Construct Email Object ---
    email_obj = Email(
        message_id=msg_id,  # Convert to string to match Pydantic model
        date=date,
        subject=subject,
        from_address=from_addr,
        to_addresses=to_addresses,
        cc_addresses=cc_addresses,
        bcc_addresses=bcc_addresses,
        body=body,
        file_name=file_name,
    )

    return email_obj


# TODO: not use this now
async def judge_correctness(
    answer: str, query: Query, judge_model: openai.OpenAI,
) -> dict:
    """Use an LLM to decide whether *answer* matches *query.answer*.

    Returns a structured ``AnswerJudgeResponse`` with a free-form *thinking*
    string (useful for debugging) and a boolean *accept* flag used for
    scoring.
    """

    system_prompt = """You are given a question, the reference answer (labelled **Reference answer**), and an answer generated by an AI assistant (labelled **AI answer**).

Follow these steps to decide whether the AI answer should be accepted:
1. Identify EXACTLY what information the **question** is asking for (e.g. who, what, when, where, why, how, quantity, etc.).
2. From the **Reference answer**, extract ONLY the facts that are required to directly satisfy the information need identified in step 1. Treat all other facts as non-essential context.
3. Verify that every essential fact from step 2 appears in the **AI answer** with the same meaning. Differences in wording, order, or additional non-conflicting details are allowed.
4. If any essential fact is missing or contradicted in the **AI answer**, then *accept* must be **false**. Otherwise *accept* must be **true**.

Important: Do NOT penalise the **AI answer** for omitting non-essential facts that appear in the **Reference answer**. The answer should only be rejected for errors or omissions in the information explicitly requested by the question.

Return your judgement as **pure JSON** (no markdown) with this exact schema:
{
  "thinking": string,  // Brief explanation of your reasoning.
  "accept": boolean   // true if the AI answer should be accepted.
}
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Question: {query.question}\n"
                f"Reference answer: {query.answer}\n"
                f"AI answer: {answer}"
            ),
        },
    ]

    response = acompletion(
        model=judge_model,
        messages=messages,
        caching=True,
        response_format=CorrectnessJudgeResponse, # TODO
    )

    first_choice = response.choices[0]  # type: ignore[attr-defined]
    raw_content = first_choice.message.content or "{}"  # type: ignore[attr-defined]

    try:
        return CorrectnessJudgeResponse.model_validate_json(raw_content)
    except Exception as e:
        # If parsing fails, fall back to 'accept': False
        return dict(
            thinking=f"Parse error: {e}\nRaw: {raw_content}", accept=False
        )


def get_rubric_state() -> dict[str, Any]:
    """
    从 rubric.jsonl 文件中读取最新的状态。
    如果文件不存在或为空，则返回一个默认的初始状态。
    """
    # 默认状态
    default_state = {"ever_found_right_email": False}
    
    if not os.path.exists(RUBRIC_FILE):
        return default_state

    try:
        with open(RUBRIC_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 过滤掉空行
            non_empty_lines = [line for line in lines if line.strip()]
            if not non_empty_lines:
                return default_state
            # 最后一行即为最新状态
            last_line = non_empty_lines[-1]
            return json.loads(last_line)
    except (json.JSONDecodeError, IndexError):
        # 如果文件损坏或格式错误，也返回默认状态
        return default_state

def save_rubric_state(state: dict[str, Any]):
    """
    将一个新的状态对象作为新行追加到 rubric.jsonl 文件中。
    """
    with open(RUBRIC_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(state) + '\n')
