"""Mock SQL execution tool for testing without database."""

import pandas as pd
from typing import Dict, Any, Optional, Annotated
from autogen_core.tools import FunctionTool


async def mock_sql_execution(
    sql: Annotated[str, "SQL query string (used for lightweight filtering in mock mode)"],
    db_url: Annotated[str, "Database URL (ignored in mock mode)"] = "mock://",
    limit: Annotated[int, "Max rows to return"] = 100,
    params: Annotated[Optional[Dict[str, Any]], "Bind parameters (optional in mock mode)"] = None,
    explain: Annotated[bool, "Return execution plan (unsupported in mock mode)"] = False,
) -> Dict[str, Any]:
    """
    Mock SQL execution returning hardcoded shipment data.
    Enhancement: If the SQL text (or params) clearly specifies one or more SHIPMENT_NUMBER values
    and/or a VOY_STOP_PORT_CDE, apply in-memory filtering so tests more closely reflect real queries.

    Supported lightweight patterns inside `sql`:
      - SHIPMENT_NUMBER = '2157351431'
      - SHIPMENT_NUMBER IN ('2157351430','2157351431')
      - VOY_STOP_PORT_CDE = 'SIN'
    We do NOT attempt full SQL parsing; just simple regex extraction.
    """

    mock_data = {
        "SHIPMENT_NUMBER": ["2157351430", "2157351431", "2157351432"],
        "SVVD": ["WM2-CVF-004 W", "WM2-CVF-004 W", "WM2-CVF-005 E"],
        "VOY_STOP_PORT_CDE": ["SIN", "SIN", "BCN"],
        "NEXT_POD": ["BCN03", "HKG01", "SIN02"],
        "SVC_TEU": [1.0, 2.0, 1.0],
        "WEIGHT_TON": [24, 30, 20],
        "WEIGHT_KG": [24000, 30000, 20000],
        "ALLOC_TCR_REGN": ["VND", "HKG", "GER"],
        "CGO_TRADE": ["AET", "AET", "IET"],
        "OP_TRADE": ["AET", "AET", "IET"],
        "CGO_NATURE": ["GC", "GC", "DG"],
        "CARGO_NATURE_GROUP": ["Dry", "Dry", "PCT"],
        "BIZ_NATURE": ["Committed", "Open", "Committed"],
        "VOY_STOP_BERTH_ARR_TM_LOC": ["23-APR-25 02.05.35 PM", "24-APR-25 08.15.00 AM", "25-APR-25 10.30.00 AM"],
        "VOY_STOP_BERTH_DEP_TM_LOC": ["24-APR-25 04.15.39 PM", "25-APR-25 06.45.00 PM", "26-APR-25 12.00.00 PM"],
    }

    df = pd.DataFrame(mock_data)

    import re
    shipment_numbers: list[str] = []
    port_code: Optional[str] = None

    if sql:
        # Extract single equality SHIPMENT_NUMBER = '123456789'
        for m in re.finditer(r"SHIPMENT_NUMBER\s*=\s*'?(\d{6,})'?,?", sql, re.IGNORECASE):
            shipment_numbers.append(m.group(1))
        # Extract IN list SHIPMENT_NUMBER IN ('123','456')
        in_match = re.search(r"SHIPMENT_NUMBER\s+IN\s*\(([^)]+)\)", sql, re.IGNORECASE)
        if in_match:
            parts = re.findall(r"'(\d{6,})'", in_match.group(1))
            shipment_numbers.extend(parts)
        # Extract port VOY_STOP_PORT_CDE = 'SIN'
        port_match = re.search(r"VOY_STOP_PORT_CDE\s*=\s*'([A-Z]{3})'", sql, re.IGNORECASE)
        if port_match:
            port_code = port_match.group(1).upper()

    # Also allow params dict to specify filters if provided
    if params:
        pn = params.get("shipment_number") or params.get("SHIPMENT_NUMBER")
        if isinstance(pn, (list, tuple)):
            shipment_numbers.extend([str(x) for x in pn])
        elif isinstance(pn, (str, int)):
            shipment_numbers.append(str(pn))
        pcode = params.get("port") or params.get("VOY_STOP_PORT_CDE")
        if isinstance(pcode, str) and len(pcode) == 3:
            port_code = pcode.upper()

    # Deduplicate
    shipment_numbers = list(dict.fromkeys(shipment_numbers))

    if shipment_numbers:
        df = df[df["SHIPMENT_NUMBER"].isin(shipment_numbers)]
    if port_code:
        df = df[df["VOY_STOP_PORT_CDE"].str.upper() == port_code]

    # Apply limit last
    if limit > 0:
        df = df.head(limit)

    result = {
        "columns": list(df.columns),
        "rows": df.values.tolist(),
        "row_count": len(df),
        "truncated": False,
        "dialect": "mock",
        "sql": sql,
        "explain": None,
        "error": None,
        "filters_applied": {
            "shipment_numbers": shipment_numbers,
            "port_code": port_code,
        },
    }
    return result


def create_mock_sql_tool() -> FunctionTool:
    """Create and return the mock SQL execution tool."""
    return FunctionTool(
        mock_sql_execution,
        name="sql_execution",
        description="Execute SQL queries (MOCK MODE: returns hardcoded data for testing)."
    )
