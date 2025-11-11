from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Awaitable, List, Dict, Tuple
from pydantic import BaseModel, Field
from datetime import datetime

### ------------------------------------- ####

key_map = {
    "p_target_svvd": "Target SVVD",
    "p_target_port_list": "Target Port List",
    "p_specified_orig_svvd": "Specified Origin SVVD",
    "p_orig_next_leg_dsch_port_cde_list": "Original Next Leg Discharge Port Code",
    "p_cargo_trade_list": "Cargo Trade",
    "p_op_trade_list": "Operating Trade",
    "p_tcr_list": "TCR",
    "p_bkg_status_list": "Booking Status",
    "p_cntr_status_list": "Container Status",
    "p_cgo_nature_list": "Cargo Nature",
    "p_cgo_nature_group_list": "Cargo Nature Group",
    "p_ccp_sales_regn_list": "CCP Sales Region",
    "p_biz_nature_list": "Business Nature",
    "p_prev_dsch_vs": "Previous Discharge VS",
    "p_orig_load_vs": "Original Load VS",
    "p_orig_dsch_vs": "Original Discharge VS",
    "p_tgt_load_vs": "Target Load VS",
    "p_tgt_dsch_vs": "Target Discharge VS",
    "p_connecting_load_vs": "Connecting Load VS",
    "p_inc_dsch_berth_arr_tm": "Inc Discharge Berth Arrival Time",
    "p_inc_dsch_berth_arr_tm_operator": "Inc Discharge Berth Arrival Time Operator",
    "p_inc_dsch_berth_dep_tm": "Inc Discharge Berth Departure Time",
    "p_inc_dsch_berth_dep_tm_operator": "Inc Discharge Berth Departure Time Operator",
    "p_orig_load_berth_arr_tm": "Original Load Berth Arrival Time",
    "p_orig_load_berth_arr_tm_operator": "Original Load Berth Arrival Time Operator",
    "p_orig_load_berth_dep_tm": "Original Load Berth Departure Time",
    "p_orig_load_berth_dep_tm_operator": "Original Load Berth Departure Time Operator",
    "p_orig_dsch_berth_arr_tm": "Original Discharge Berth Arrival Time",
    "p_orig_dsch_berth_arr_tm_operator": "Original Discharge Berth Arrival Time Operator",
    "p_orig_dsch_berth_dep_tm": "Original Discharge Berth Departure Time",
    "p_orig_dsch_berth_dep_tm_operator": "Original Discharge Berth Departure Time Operator",
    "p_tgt_load_berth_arr_tm": "Target Load Berth Arrival Time",
    "p_tgt_load_berth_arr_tm_operator": "Target Load Berth Arrival Time Operator",
    "p_tgt_load_berth_dep_tm": "Target Load Berth Departure Time",
    "p_tgt_load_berth_dep_tm_operator": "Target Load Berth Departure Time Operator",
    "p_tgt_dsch_berth_arr_tm": "Target Discharge Berth Arrival Time",
    "p_tgt_dsch_berth_arr_tm_operator": "Target Discharge Berth Arrival Time Operator",
    "p_tgt_dsch_berth_dep_tm": "Target Discharge Berth Departure Time",
    "p_tgt_dsch_berth_dep_tm_operator": "Target Discharge Berth Departure Time Operator",
    "p_tgt_next_leg_load_berth_arr_tm": "Target Next Leg Load Berth Arrival Time",
    "p_tgt_next_leg_load_berth_arr_tm_operator": "Target Next Leg Load Berth Arrival Time Operator",
    "p_tgt_next_leg_load_berth_dep_tm": "Target Next Leg Load Berth Departure Time",
    "p_tgt_next_leg_load_berth_dep_tm_operator": "Target Next Leg Load Berth Departure Time Operator",
    "p_orig_connection_time": "Original Connection Time",
    "p_orig_connection_operator": "Original Connection Time Operator",
    "p_tgt_connection_time_load": "Target Connection Time (Load)",
    "p_tgt_connection_load_operator": "Target Connection Time (Load) Operator",
    "p_tgt_connection_time_dsch": "Target Connection Time (Discharge)",
    "p_tgt_connection_dsch_operator": "Target Connection Time (Discharge) Operator",
    "p_orig_idling_day": "Original Idling Day",
    "p_orig_idling_day_operator": "Original Idling Day Operator",
    "p_tgt_idling_day": "Target Idling Day",
    "p_tgt_idling_day_operator": "Target Idling Day Operator"
}
remove_keys = {'key'}           # Remove 'remove_me' key

###

def rename_and_remove_keys(d):
    """
    Rename and remove keys in a dictionary.
    
    :param d: Original dictionary
    :param key_map: Dict mapping old keys to new keys
    :param remove_keys: Set or list of keys to remove
    :return: New dictionary with keys renamed and specified keys removed
    """
    return {
        key_map.get(k, k): v
        for k, v in d.items()
        if k not in remove_keys
    }

class LogMessage(BaseModel):
    agent_name: str
    action: str
    content: str
    is_complete: Optional[bool] = False
    references: Optional[List[Dict]] = []
    id: str
    timestamp: datetime = Field(default_factory=datetime.now)

class AgentOps(ABC):
    
    def __init__(self) -> None:
        self.on_message_cb: Optional[Callable[[LogMessage], Awaitable[None]]] = None
    
    @abstractmethod
    def run(self, *args, **kwargs) -> Tuple[Any, str]:
        pass
    
        