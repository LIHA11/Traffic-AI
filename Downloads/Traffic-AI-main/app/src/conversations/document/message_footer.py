from mongoengine import EmbeddedDocument, StringField, EnumField, ListField

from src.conversations.enum.message_footer_type_enum import MessageFooterTypeEnum


class MessageFooter(EmbeddedDocument):
    key = StringField(required=True)
    type = EnumField(MessageFooterTypeEnum, required=True)
    content = StringField(required=False)
    options = ListField(StringField(), required=False)
