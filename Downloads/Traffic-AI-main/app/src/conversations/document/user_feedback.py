from mongoengine import (
    Document,
    BooleanField,
    ReferenceField,
    DateTimeField,
    StringField,
)
from datetime import datetime, timezone


class UserFeedback(Document):
    meta = {"collection": "userFeedbacks", "indexes": ["message"], "strict": False}
    message = ReferenceField("Message", required=True, unique=True)
    liked = BooleanField(required=True)
    feedback = StringField(null=True)
    summary = StringField(null=True)
    created_date_time_utc = DateTimeField(
        db_field="createdDateTimeUtc",
    )
    last_modified_date_time_utc = DateTimeField(
        db_field="lastModifiedDateTimeUtc",
    )

    def save(self, *args, **kwargs):
        if not self.created_date_time_utc:
            self.created_date_time_utc = datetime.now(tz=timezone.utc)
        self.last_modified_date_time_utc = datetime.now(tz=timezone.utc)
        return super(UserFeedback, self).save(*args, **kwargs)
