class BreRequestDto:
    shipment_numbers: list[str]
    original_leg_load_stop_id: str
    original_leg_dsch_stop_id: str
    new_leg_load_stop_id: str
    new_leg_dsch_stop_id: str

    def __init__(
        self,
        shipment_numbers: list[str],
        original_leg_load_stop_id: str,
        original_leg_dsch_stop_id: str,
        new_leg_load_stop_id: str,
        new_leg_dsch_stop_id: str,
    ):
        self.shipment_numbers = shipment_numbers
        self.original_leg_load_stop_id = original_leg_load_stop_id
        self.original_leg_dsch_stop_id = original_leg_dsch_stop_id
        self.new_leg_load_stop_id = new_leg_load_stop_id
        self.new_leg_dsch_stop_id = new_leg_dsch_stop_id

    def to_dict(self):
        return {
            "shipmentNumbers": self.shipment_numbers,
            "originalLegLoadStopId": self.original_leg_load_stop_id,
            "originalLegDschStopId": self.original_leg_dsch_stop_id,
            "newLegLoadStopId": self.new_leg_load_stop_id,
            "newLegDschStopId": self.new_leg_dsch_stop_id,
        }
