from dataclasses import dataclass

from src.conversations.dto.message_footer_dto import MessageFooterDto


@dataclass
class PresetMessageRequestDto:
    request_content: str
    preset_system_content: str
    preset_system_footer: MessageFooterDto

    @classmethod
    def from_dict(cls, data: dict) -> "PresetMessageRequestDto":
        return cls(
            request_content=data.get("requestContent"),
            preset_system_content=data.get("presetSystemContent"),
            preset_system_footer=(
                MessageFooterDto.from_dict(data.get("presetSystemFooter"))
                if data.get("presetSystemFooter")
                else None
            ),
        )
