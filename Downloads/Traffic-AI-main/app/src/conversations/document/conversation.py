from mongoengine import (
    Document,
    StringField,
    DateTimeField,
    ListField,
    ReferenceField,
    BooleanField,
)
from datetime import datetime, timezone
from src.conversations.document.message import Message


class Conversation(Document):
    meta = {"collection": "conversations", "indexes": ["creator"], "strict": False}
    creator = StringField(required=True)
    messages = ListField(ReferenceField(Message), default=[])
    is_deleted = BooleanField(db_field="isDeleted", default=False)
    created_date_time_utc = DateTimeField(
        db_field="createdDateTimeUtc",
    )
    last_modified_date_time_utc = DateTimeField(
        db_field="lastModifiedDateTimeUtc",
    )

    def delete(self, *args, **kwargs):
        for message in self.messages:
            message.delete()

        super().delete(*args, **kwargs)

    def save(self, *args, **kwargs):
        if not self.created_date_time_utc:
            self.created_date_time_utc = datetime.now(tz=timezone.utc)
        self.last_modified_date_time_utc = datetime.now(tz=timezone.utc)
        return super(Conversation, self).save(*args, **kwargs)
