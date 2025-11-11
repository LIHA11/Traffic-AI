"""Mock SQL execution tool for testing without database."""

import pandas as pd
from typing import Dict, Any, Optional, Annotated
from autogen_core.tools import FunctionTool


async def mock_sql_execution(
    sql: Annotated[str, "SQL query string (ignored in mock mode)"],
    db_url: Annotated[str, "Database URL (ignored in mock mode)"] = "mock://",
    limit: Annotated[int, "Max rows to return"] = 100,
    params: Annotated[Optional[Dict[str, Any]], "Bind parameters"] = None,
    explain: Annotated[bool, "Return execution plan"] = False,
) -> Dict[str, Any]:
    """
    Mock SQL execution that returns hardcoded shipment data.
    Useful for testing downstream agents without database connection.
    """
    
    # 硬编码的示例数据 - 可以根据需要修改
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
    
    # 应用 limit
    if limit > 0:
        df = df.head(limit)
    
    # 返回格式需要与真实 SQL 工具一致
    result = {
        "columns": list(df.columns),
        "rows": df.values.tolist(),  # 转换为嵌套列表格式
        "row_count": len(df),
        "truncated": False,
        "dialect": "mock",
        "sql": sql,
        "explain": None,
        "error": None,
    }
    
    return result


def create_mock_sql_tool() -> FunctionTool:
    """Create and return the mock SQL execution tool."""
    return FunctionTool(
        mock_sql_execution,
        name="sql_execution",
        description="Execute SQL queries (MOCK MODE: returns hardcoded data for testing)."
    )
