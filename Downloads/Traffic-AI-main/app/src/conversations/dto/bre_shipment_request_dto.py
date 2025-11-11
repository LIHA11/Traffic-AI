from dataclasses import dataclass
from typing import List


@dataclass
class BreShipmentDto:
    shipment_number: str
    orig_next_seg_load_id: str
    orig_next_seg_dsch_id: str
    tgt_next_seg_load_id: str
    tgt_next_seg_dsch_id: str

    @classmethod
    def from_dict(cls, data: dict) -> "BreShipmentRequestDto":
        return cls(
            shipment_number=data.get("shipmentNumber"),
            orig_next_seg_load_id=data.get("origNextSegLoadId"),
            orig_next_seg_dsch_id=data.get("origNextSegDschId"),
            tgt_next_seg_load_id=data.get("tgtNextSegLoadId"),
            tgt_next_seg_dsch_id=data.get("tgtNextSegDschId"),
        )


@dataclass
class BreShipmentRequestDto:
    shipments: List[BreShipmentDto]

    @classmethod
    def from_dict(cls, data: dict) -> "BreShipmentRequestDto":
        shipments_data = data.get("shipments", [])
        shipments = [BreShipmentDto.from_dict(shipment) for shipment in shipments_data]
        return cls(shipments=shipments)
