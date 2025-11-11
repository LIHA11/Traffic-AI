from mongoengine import (
    Document,
    StringField,
    EnumField,
    DateTimeField,
    DictField,
    ReferenceField,
    EmbeddedDocumentField,
)

from src.conversations.document.message_footer import MessageFooter
from src.conversations.enum.message_type_enum import MessageTypeEnum
from src.conversations.enum.role_enum import RoleEnum
from src.conversations.document.user_feedback import UserFeedback
from datetime import datetime, timezone


class Message(Document):
    meta = {"collection": "messages", "strict": False}
    role = EnumField(RoleEnum, required=True)
    type = EnumField(
        MessageTypeEnum, required=True, default=MessageTypeEnum.CONVERSATION
    )
    content = StringField(required=True)
    footer = EmbeddedDocumentField(MessageFooter, required=False)
    userFeedback = ReferenceField(UserFeedback)
    conversation = ReferenceField("Conversation", required=True)
    timestamp = DateTimeField(required=True)
    group_id = StringField(db_field="groupId", required=False)
    metadata = DictField(default={})
    others = DictField(default={})
    created_date_time_utc = DateTimeField(
        db_field="createdDateTimeUtc",
    )
    last_modified_date_time_utc = DateTimeField(
        db_field="lastModifiedDateTimeUtc",
    )

    def delete(self, *args, **kwargs):
        if self.userFeedback:
            self.userFeedback.delete()

        super().delete(*args, **kwargs)

    def save(self, *args, **kwargs):
        if not self.created_date_time_utc:
            self.created_date_time_utc = datetime.now(tz=timezone.utc)
        self.last_modified_date_time_utc = datetime.now(tz=timezone.utc)
        return super(Message, self).save(*args, **kwargs)
